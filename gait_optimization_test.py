import mujoco
import mujoco.viewer
import time
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class gait_optimization:
    '''
    Position control parameters:
    theta_desired(t) = a_i + b_i * sin(omega*t + c_i)
    
    [[a_0, b_0, c_0],
     [a_1, b_1, c_1],
     [a_2, b_2, c_2],
     [a_3, b_3, c_3]]
    
    where 0 is left shoulder, 1 is left elbow, 2 is right shoulder and 3 is right elbow
    
    limits (based on joint ranges in MJCF):
    -0.57 <= a - b <= a + b <= 0.57
    (ensuring the sinusoidal trajectory stays within joint limits)
    0 <= c <= 2*pi

    axes:
    x is 'out of the page', y is to the right and z is upwards
    '''   
    def __init__(self, num_params, hill_climber_noise_std):
        self.num_params = num_params
        self.hill_climber_noise_std = hill_climber_noise_std
        self.frequency = 3  # Controls the speed of the oscillation (Hz for gait pattern)
        self.servo_update_rate = 10  # Servos update at 10 Hz
        self.servo_update_period = 1.0 / self.servo_update_rate  # 0.1 seconds between updates
        
    def create_initial_params(self):
        # Start with more aggressive oscillations for better locomotion
        params = [
            [0.1, 0.3, 0],      # Left shoulder - forward bias
            [0.0, 0.25, 0],     # Left elbow
            [-0.1, 0.3, 0],     # Right shoulder - backward bias
            [0.0, 0.25, 0]      # Right elbow
        ]
        return params

    def evaluate_policy(self, model, data, params):
        # Set stable initial configuration
        data.qpos[0:len(params)] = [0.0, 0.0, 0.0, 0.0]
        data.ctrl[:] = 0.0
        
        # Settle the robot
        for _ in range(100):
            mujoco.mj_step(model, data)
        
        start_pos = data.body('base_mesh').xpos.copy()
        
        # Calculate number of steps for 10 seconds of sim time
        sim_duration = 10.0
        num_steps = int(sim_duration / model.opt.timestep)
        
        last_servo_update = 0.0
        
        # Run for exactly num_steps
        for _ in range(num_steps):
            # Only update servo commands at 10 Hz (every 0.1 seconds)
            if data.time - last_servo_update >= self.servo_update_period:
                # Apply desired position commands (position control)
                data.ctrl[0] = params[0][0] + (params[0][1] * math.sin((self.frequency * data.time) + params[0][2]))
                data.ctrl[1] = params[1][0] + (params[1][1] * math.sin((self.frequency * data.time) + params[1][2]))
                data.ctrl[2] = params[2][0] + (params[2][1] * math.sin((self.frequency * data.time) + params[2][2]))
                data.ctrl[3] = params[3][0] + (params[3][1] * math.sin((self.frequency * data.time) + params[3][2]))
                
                last_servo_update = data.time
            
            mujoco.mj_step(model, data)
        
        # Use distance travelled as reward
        new_pos = data.body('base_mesh').xpos
        reward = np.linalg.norm(new_pos - start_pos)
        return reward
    
    def generate_random_search_params(self):
        # Generate a new, random set of parameters ensuring they stay within joint limits
        n = 4
        candidate_params = []
        
        for _ in range(n):
            # Generate offset 'a' within safe range (allow more range for aggressive motion)
            a = np.random.uniform(-0.4, 0.4)
            
            # Generate amplitude 'b' ensuring a-b >= -0.57 and a+b <= 0.57
            max_amplitude = min(0.57 - a, 0.57 + a)
            b = np.random.uniform(0.1, max_amplitude)  # Minimum amplitude of 0.1 for movement
            
            # Generate phase 'c'
            c = np.random.uniform(0, 2 * np.pi)
            
            candidate_params.append([a, b, c])
        
        return np.array(candidate_params)
    
    def generate_hill_climber_params(self, best_params):
        # Generate a small random "step" (noise)
        noise = np.random.normal(
            loc=0.0,  # mean
            scale=self.hill_climber_noise_std,  # standard deviation (step size)
            size=self.num_params
        )
        
        candidate_params = best_params + noise
        
        # Reshape to (n, 3) for constraint application
        candidate_params = candidate_params.reshape(-1, 3)

        # Apply constraints to ensure trajectory stays within joint limits [-0.57, 0.57]
        for i in range(len(candidate_params)):
            a = candidate_params[i, 0]
            b = candidate_params[i, 1]
            
            # Clip offset to safe range
            a = np.clip(a, -0.4, 0.4)
            
            # Ensure amplitude keeps trajectory within joint limits
            max_amplitude = min(0.57 - abs(a), 0.57)
            b = np.clip(b, 0, max_amplitude)
            
            # Clip phase
            c = np.clip(candidate_params[i, 2], 0, 2 * np.pi)
            
            candidate_params[i] = [a, b, c]
        
        return candidate_params

    def run_simple_optimization(self, num_iterations, model, data, method):
        # Initialization
        best_params = self.create_initial_params()
        best_reward = self.evaluate_policy(model, data, best_params)
        print(f"Initial policy reward: {best_reward}")
        reward_history = [best_reward]
        
        # Optimization
        for iteration in range(num_iterations):
            print(f"iteration {iteration}")
            
            if method == "random search":
                candidate_params = self.generate_random_search_params()
                
            elif method == "hill climber":
                candidate_params = self.generate_hill_climber_params(best_params)
            
            # Evaluation
            candidate_reward = self.evaluate_policy(model, data, candidate_params)
            
            # Update
            if candidate_reward > best_reward:
                best_reward = candidate_reward
                best_params = candidate_params
                print(f" Iteration {iteration + 1}/{num_iterations} | New Best Reward (climbed): {best_reward}")
            
            # Record current best reward for learning curve plot
            reward_history.append(best_reward)
        
        print(f"Final best parameters: {best_params}")
        print(f"Final best reward: {best_reward}")
        return best_params, reward_history
    
    def parallel_random_mutation_hill_climber(self, num_iterations, model, data, n, param_list=[], reward_history=[]):
        if num_iterations == 0:
            print(f"Final best parameters: {param_list[0][0]}")
            print(f"Final best reward: {param_list[0][1]}")
            return param_list[0][0], reward_history
        else:
            print(f"iteration {num_iterations}")
            
            # generate initial n random params on first run
            if param_list == []:
                for _ in range(n):
                    params = self.generate_random_search_params()
                    reward = self.evaluate_policy(model, data, params)
                    param_list.append((params, reward))
            
            # remake list of just parameters in non-start case
            params = [param for param, reward in param_list]
            
            # add a mutation for each set of parameters
            for i in range(n):
                mutated_params = self.generate_hill_climber_params(params[i])
                mutated_reward = self.evaluate_policy(model, data, mutated_params)
                
                # store params with their calculated rewards
                param_list.append((mutated_params, mutated_reward))
            
            # sort list of length 2n by reward
            param_list.sort(reverse=True, key=lambda x: x[1])
            print("rewards of sorted param list")
            for (params, reward) in param_list:
                print(reward)
            
            # remove the bottom half
            half_index = len(param_list) // 2
            upper_half = param_list[:half_index]
            
            # store current best reward (for reward history plot)
            reward_history.append(upper_half[0][1])
            
            # recurse
            return self.parallel_random_mutation_hill_climber(num_iterations - 1, model, data, n, upper_half, reward_history)
    
    def plot_reward_progress(self, reward_history):
        """
        Plots the progress of rewards over iterations.
        """
        iterations = range(len(reward_history))

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, reward_history, marker='o', linestyle='-', color='blue')

        plt.title('Progress of Reward Over Iterations (Position Control)')
        plt.xlabel('Iterations')
        plt.ylabel('Reward')

        plt.grid()

        plt.savefig("learning_curve_position.png")
        print("Saved learning_curve_position.png")
        
    def plot_reward_histories_with_errors(self, reward_histories):
        """
        Plots multiple reward histories with error bars representing the standard deviation.
        """
        rewards_array = np.array(reward_histories)
        
        means = np.mean(rewards_array, axis=0)
        std_devs = np.std(rewards_array, axis=0)

        iterations = range(len(means))

        plt.figure(figsize=(12, 8))
        plt.errorbar(iterations, means, yerr=std_devs, marker='o', linestyle='-', capsize=5, color='blue')

        plt.title('Progress of Reward Over Iterations with Error Bars (Position Control)')
        plt.xlabel('Iterations')
        plt.ylabel('Reward')

        plt.grid()
        
        plt.savefig("learning_curve_multi_position.png")
        print("Saved learning_curve_multi_position.png")
        

def main():
    xml_path = "asset/myrobot_test.mjcf"

    # Load the model from the specified XML file.
    model = mujoco.MjModel.from_xml_path(xml_path)

    # The MjData object contains the dynamic state of the simulation
    data = mujoco.MjData(model)
    
    num_params = (4, 3)
    hill_climber_noise_std = 0.05  # Smaller noise for position control
    optim = gait_optimization(num_params, hill_climber_noise_std)
    
    num_iterations = 1000
    final_params, reward_history = optim.run_simple_optimization(num_iterations, model, data, "hill climber")
    #final_params, reward_history = optim.parallel_random_mutation_hill_climber(num_iterations, model, data, n=10)
    optim.plot_reward_progress(reward_history)
    
if __name__ == "__main__":
    main()