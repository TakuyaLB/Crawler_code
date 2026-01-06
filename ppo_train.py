"""
ppo_train.py

Train PPO to predict next actuator positions for the 4 actuators.
Observation: [roll, pitch, yaw, qpos_joint0, qpos_joint1, qpos_joint2, qpos_joint3]
Action: 4 continuous values = next desired absolute joint positions (radians)

The environment steps the physics for one servo update period (0.1s) after applying the action,
and returns reward equal to forward displacement of body('base_mesh') during that period.

Based on the gait_optimization_test.py logic.
"""
import gymnasium as gym
import numpy as np
import mujoco
import math
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback

XML_PATH = "asset/myrobot_test.mjcf"
episode_rewards = []
best_reward_history = []
best_reward_so_far = -1e9

class RewardTrackingCallback(BaseCallback):
    """
    Tracks per-episode reward and best reward so far, without interfering
    with EvalCallback or SB3 rollout.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        global episode_rewards, best_reward_history, best_reward_so_far

        infos = self.locals.get("infos", None)
        if infos is None:
            return True

        # Check if an episode finished (Monitor adds "episode" to info)
        if "episode" in infos[0]:
            ep_reward = infos[0]["episode"]["r"]
            episode_rewards.append(ep_reward)

            if ep_reward > best_reward_so_far:
                best_reward_so_far = ep_reward

            best_reward_history.append(best_reward_so_far)

        return True

# --- helper: quaternion -> euler (roll, pitch, yaw) ---
def quat_to_euler(q):
    # q is [w, x, y, z]
    w, x, y, z = q
    # roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    # pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    # yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return np.array([roll, pitch, yaw], dtype=np.float32)


class MujocoGaitEnv(gym.Env):
    """
    Gym environment wrapping your MuJoCo setup.
    Each action is the desired absolute joint positions for 4 actuators.
    Observation contains roll,pitch,yaw and the current 4 joint positions.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, xml_path=XML_PATH, servo_update_rate=10, episode_servo_steps=100):
        super().__init__()

        # Load model + data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # servo settings copied from your script
        self.frequency = 3.0
        self.servo_update_rate = servo_update_rate
        self.servo_update_period = 1.0 / float(self.servo_update_rate)
        self.episode_servo_steps = episode_servo_steps  # number of servo updates per episode (e.g., 100 -> 10s)

        # action: absolute positions for 4 actuators
        self.action_space = spaces.Box(low=-0.57, high=0.57, shape=(4,), dtype=np.float32)

        # observation: roll,pitch,yaw (3) + 4 joint positions = 7
        obs_low = np.array([-np.pi, -np.pi / 2, -np.pi] + [-0.57] * 4, dtype=np.float32)
        obs_high = np.array([np.pi, np.pi / 2, np.pi] + [0.57] * 4, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # internal counters
        self.current_servo_step = 0

        # initialize reset state (same as your script)
        self._init_qpos = None
        self._prepare_initial_configuration()

    def _prepare_initial_configuration(self):
        # Prepare qpos baseline used in reset (match your original initialization)
        # First 7 DOFs are freejoint (3 pos, 4 quat), then 4 joint angles
        qpos = np.zeros(self.model.nq, dtype=np.float64)
        # place slightly above ground
        if len(qpos) >= 3:
            qpos[2] = 0.1
        # quaternion no rotation
        if len(qpos) >= 7:
            qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        # set initial actuator positions to zeros (or small offsets)
        if len(qpos) >= 11:
            qpos[7:7+4] = [0.0, 0.0, 0.0, 0.0]
        self._init_qpos = qpos

    def reset(self, *, seed=None, options=None):
        # Seeding
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        
        # Reset model state
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = self._init_qpos.copy()
        # set ctrl initial values = qpos joints
        for j in range(4):
            self.data.ctrl[j] = float(self.data.qpos[7 + j])

        self.current_servo_step = 0

        # settle the robot a bit (as in your script)
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        info = {}  # You can return extra info or leave it empty
        
        return self._get_obs(), info

    def _get_obs(self):
        # read quaternion from qpos[3:7] -> convert to euler
        quat = self.data.qpos[3:7].copy()  # [w,x,y,z]
        euler = quat_to_euler(quat)
        joint_pos = self.data.qpos[7:7+4].astype(np.float32)
        obs = np.concatenate([euler.astype(np.float32), joint_pos])
        return obs

    def step(self, action):
        """
        Apply action, simulate for the servo interval,
        return Gymnasium-style: obs, reward, terminated, truncated, info
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Save position & height
        start_pos = self.data.body('base_mesh').xpos.copy()
        start_z = start_pos[2]

        # Apply action (desired joint positions)
        self.data.ctrl[0:4] = action.astype(np.float64)

        # Simulate for servo interval
        n_steps = int(math.ceil(self.servo_update_period / self.model.opt.timestep))
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

        # Final position
        end_pos = self.data.body('base_mesh').xpos.copy()
        end_z = end_pos[2]

        # -----------------------
        #  REWARD FUNCTION
        # -----------------------

        # 1) Forward displacement (main reward)
        forward_reward = float(np.linalg.norm(end_pos - start_pos))

        # 2) Jumping penalty (upward movement)
        vertical_jump = max(0.0, end_z - start_z)
        jumping_penalty = 2.0 * vertical_jump

        # 3) Joint velocity smoothness penalty
        joint_vel_penalty = 0.02 * np.sum(np.abs(self.data.qvel[6:10]))

        # 4) Actuator force jerk penalty
        if not hasattr(self, "prev_force"):
            self.prev_force = np.zeros(4)

        force_diff = np.sum(np.abs(self.data.actuator_force[:4] - self.prev_force))
        force_penalty = 0.01 * force_diff
        self.prev_force = self.data.actuator_force[:4].copy()

        # Total reward
        reward = (
            forward_reward
            - (0.2 * jumping_penalty)
            - (0.2 * joint_vel_penalty)
            - (0.2 * force_penalty)
        )

        # -----------------------

        self.current_servo_step += 1
        terminated = False
        truncated = self.current_servo_step >= self.episode_servo_steps

        obs = self._get_obs()
        info = {"time": float(self.data.time)}

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        # User can launch their own viewer for testing; training will be headless.
        pass

    def close(self):
        pass

def plot_best_reward_curve():
    if len(best_reward_history) == 0:
        print("No reward data collected — skipping plot.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(best_reward_history, label="Best Reward So Far", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress – Best Reward Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("best_reward_curve.png")
    print("Saved best_reward_curve.png")

def train_model(
    timesteps=200_000,
    model_save_path="ppo_gait_actuator_1.zip",
    xml_path=XML_PATH,
    verbose=1,
):
    env = MujocoGaitEnv(xml_path)
    # Optional env check
    try:
        check_env(env)
        print("Environment check passed.")
    except Exception as e:
        print("Environment check warning (may be fine):", e)

    model = PPO("MlpPolicy", env, verbose=verbose, learning_rate=3e-4, n_steps=2048, batch_size=64, gamma=0.99)
    eval_env = MujocoGaitEnv(xml_path)
    eval_callback = EvalCallback(eval_env, best_model_save_path="./ppo_best_model/", log_path="./ppo_logs/",
                                 eval_freq=5000, deterministic=True)
    
    reward_callback = RewardTrackingCallback()

    callback = CallbackList([eval_callback, reward_callback])

    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    model.save(model_save_path)
    print(f"Saved model to {model_save_path}")
    
    # Plot best reward graph
    plot_best_reward_curve()

    return model


if __name__ == "__main__":
    # quick train example
    train_model(timesteps=200_000)
