import time
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from ppo_train import MujocoGaitEnv


def main():
    model_path = "ppo_gait_actuator_1.zip"

    env = MujocoGaitEnv()
    model = PPO.load(model_path)

    obs, _ = env.reset()

    mj_model = env.model
    mj_data = env.data

    # This is the TRUE control period
    control_dt = env.servo_update_period

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        print("Viewer launched. Running PPO policy...")

        last_time = time.time()

        while viewer.is_running():
            # Policy inference
            action, _ = model.predict(obs, deterministic=True)

            # Step environment (internally advances physics)
            obs, reward, terminated, truncated, info = env.step(action)

            # Sync viewer
            viewer.sync()

            # Real-time pacing (MATCH TRAINING)
            now = time.time()
            elapsed = now - last_time
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.time()

            if terminated or truncated:
                obs, _ = env.reset()


if __name__ == "__main__":
    main()
