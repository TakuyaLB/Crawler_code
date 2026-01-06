"""
ppo_test.py

Load trained PPO model and run inference in the same viewer loop style as mesh_demo_test.py.
At each servo update (10 Hz) the PPO model predicts the next absolute positions for the 4 actuators.
The rest of the script logs joint positions, desired positions, speeds, actuator forces and saves plots
like the original mesh_demo_test.py.
"""
import mujoco
import mujoco.viewer
import time
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from ppo_train import quat_to_euler, MujocoGaitEnv, XML_PATH  # reuse helper and xml_path

MODEL_PATH = "ppo_gait_actuator_1.zip"  # adjust if you saved elsewhere

def run_test(model_path=MODEL_PATH, render=True):#, sim_seconds=10.0):
    # Load model & environment
    model = PPO.load(model_path)

    # load model/data for viewer
    model_mjc = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model_mjc)

    # copied init from your script
    # set initial freejoint and actuator qpos and ctrl
    data.qpos[2] = 0.1
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:11] = [0.0, 0.0, 0.0, 0.0]
    data.ctrl[:] = [0.0, 0.0, 0.0, 0.0]

    # settle
    for _ in range(1000):
        mujoco.mj_step(model_mjc, data)

    # logging buffers (match your mesh_demo_test.py layout)
    num_joints = 4
    time_log = []
    q_log = [[] for _ in range(num_joints)]
    qd_log = [[] for _ in range(num_joints)]
    q_desired_log = [[] for _ in range(num_joints)]
    actuator_force_log = [[] for _ in range(num_joints)]

    # viewer
    with mujoco.viewer.launch_passive(model_mjc, data, show_left_ui=True, show_right_ui=True) as viewer:
        last_servo_update = -1e9
        # run until sim_seconds elapsed or viewer closed
        while viewer.is_running():# and (data.time < sim_seconds):
            step_start = time.time()

            # At servo update times, query the policy and write ctrl
            if data.time - last_servo_update >= (1.0 / 10.0):
                # build observation like training env: roll,pitch,yaw and current 4 joint qpos
                quat = data.qpos[3:7]
                euler = quat_to_euler(quat)
                joint_pos = data.qpos[7:11].astype(np.float32)
                obs = np.concatenate([euler.astype(np.float32), joint_pos]).astype(np.float32)
                
                print(f"input: {obs}")
                
                action, _ = model.predict(obs, deterministic=True)
                
                # In ppo_test.py, after getting the action from the model
                print(f"Action from model (rad): {action}")
                print(f"Action from model (deg): {action * 180/np.pi}")
                
                # apply action to ctrl (action interpreted as absolute desired positions)
                data.ctrl[0:4] = action.astype(np.float64)

                last_servo_update = data.time

            # logging
            time_log.append(data.time)
            for j in range(num_joints):
                q_log[j].append(data.qpos[7 + j])
                # qvel: freejoint velocities occupy 6 first entries; joint velocities start at index 6
                qd_log[j].append(data.qvel[6 + j])
                q_desired_log[j].append(data.ctrl[j])
                # actuator force is available as data.actuator_force
                actuator_force_log[j].append(data.actuator_force[j] if j < len(data.actuator_force) else 0.0)

            # step sim
            mujoco.mj_step(model_mjc, data)
            viewer.sync()

            # keep real-time-ish
            time_until_next_step = model_mjc.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # After viewer exits, produce plots similar to mesh_demo_test.py
    plt.figure(figsize=(10, 12))

    # Motor angles - actual vs desired
    plt.subplot(4, 1, 1)
    for j in range(num_joints):
        plt.plot(time_log, q_log[j], label=f"joint{j} actual", linestyle='-')
        plt.plot(time_log, q_desired_log[j], label=f"joint{j} desired", linestyle='--', alpha=0.7)
    plt.title("Motor Angles - Actual vs Desired (rad)")
    plt.ylabel("Angle (rad)")
    plt.grid(True)
    plt.legend()

    # Motor speeds
    plt.subplot(4, 1, 2)
    for j in range(num_joints):
        plt.plot(time_log, qd_log[j], label=f"joint{j}")
    plt.title("Motor Speeds (rad/s)")
    plt.ylabel("Speed (rad/s)")
    plt.grid(True)
    plt.legend()

    # Tracking error
    plt.subplot(4, 1, 3)
    for j in range(num_joints):
        error = np.array(q_desired_log[j]) - np.array(q_log[j])
        plt.plot(time_log, error, label=f"joint{j}")
    plt.title("Position Tracking Error (rad)")
    plt.ylabel("Error (rad)")
    plt.grid(True)
    plt.legend()

    # Actuator forces (computed by position servo)
    plt.subplot(4, 1, 4)
    for j in range(num_joints):
        plt.plot(time_log, actuator_force_log[j], label=f"joint{j}")
    plt.title("Actuator Forces (N)")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("motor_plots_position_ppo_test.png")
    print("Saved motor_plots_position_ppo_test.png")


if __name__ == "__main__":
    run_test()
