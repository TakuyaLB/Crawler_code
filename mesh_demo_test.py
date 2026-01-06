import mujoco
import mujoco.viewer
import time
import math
import numpy as np
import torch
import torch.nn as nn
from ppo_train import quat_to_euler

# Use NON-GUI backend for matplotlib (critical on macOS)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xml_path = "asset/myrobot_test.mjcf"

# Load the model from the specified XML file.
model = mujoco.MjModel.from_xml_path(xml_path)

# The MjData object contains the dynamic state of the simulation
data = mujoco.MjData(model)

# Define motion parameters
amplitude = 0.3  # Reduced for position control (in radians)
frequency = 3    # Controls the speed of the oscillation (Hz for gait pattern)
servo_update_rate = 10  # Servos update at 10 Hz
servo_update_period = 1.0 / servo_update_rate  # 0.1 seconds between updates

num_joints = 4

# Gait parameters - these now represent DESIRED POSITIONS (in radians)
# Parameters: [offset, amplitude, phase]
# More aggressive amplitudes and coordinated phases for walking
gait_params = [[ 0.01937524,  0.27656218,  0.05542189],
 [-0.08386459,  0.16256345,  0.0        ],
 [-0.28541862,  0.28458138,  0.06174749],
 [ 0.07287496,  0.28122491,  0.0        ]]


# Set stable initial configuration - start with robot on ground
# First 7 DOFs are freejoint (3 pos, 4 quat), then 4 joint angles
'''
data.qpos[2] = 0.1  # Set z-position slightly above ground
data.qpos[3:7] = [1, 0, 0, 0]  # Quaternion for no rotation
data.qpos[7:7+num_joints] = [gait_params[i][0] for i in range(num_joints)]
data.ctrl[:] = [gait_params[i][0] for i in range(num_joints)]
'''

# Logs
time_log = []
q_log = [[] for _ in range(num_joints)]
qd_log = [[] for _ in range(num_joints)]
q_desired_log = [[] for _ in range(num_joints)]
actuator_force_log = [[] for _ in range(num_joints)]

# Matplotlib setup
plt.ion()
fig, axs = plt.subplots(3, 1, figsize=(8, 10))
(fig_q, fig_qd, fig_qd_desired) = axs

fig_q.set_title("Motor Angle (rad)")
fig_qd.set_title("Motor Speed (rad/s)")
fig_qd_desired.set_title("Desired Position (rad)")

for ax in axs:
    ax.set_xlim(0, 5)  # show last 5 seconds
    ax.grid(True)

line_q, = fig_q.plot([], [], label="joint0")
line_qd, = fig_qd.plot([], [], label="joint0")
line_qd_desired, = fig_qd_desired.plot([], [], label="joint0_desired")

fig_q.legend()
fig_qd.legend()
fig_qd_desired.legend()

def update_plot():
    """Update Matplotlib plot for real-time display."""
    window = 5.0  # last 5 seconds
    t = np.array(time_log)

    t_mask = t > t[-1] - window

    line_q.set_data(t[t_mask], np.array(q_log)[t_mask])
    line_qd.set_data(t[t_mask], np.array(qd_log)[t_mask])
    line_qd_desired.set_data(t[t_mask], np.array(q_desired_log)[t_mask])

    for ax in axs:
        ax.set_xlim(max(0, t[-1] - window), t[-1])

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.001)
    
# Settle the robot (let it adapt to gravity)
print("Settling robot...")
for i in range(1000):
    # Keep servos at initial position during settling
    data.ctrl[:] = [gait_params[j][0] for j in range(num_joints)]
    mujoco.mj_step(model, data)
    
    # Check for instability
    if np.any(np.isnan(data.qpos)) or np.any(np.isinf(data.qpos)):
        print(f"ERROR: Simulation became unstable during settling at step {i}!")
        print(f"qpos: {data.qpos}")
        exit(1)

print("Settling complete. Robot stabilized.")
print("Starting control...")
print(f"Model timestep: {model.opt.timestep} seconds")
print(f"Servo update rate: {servo_update_rate} Hz (every {servo_update_period} seconds)")

# Launch the passive viewer
# Set viewer to run in real-time by default (30 FPS rendering)
with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
    
    last_servo_update = -servo_update_period  # Initialize to send command immediately
    
    # Run the simulation loop as long as the viewer window is open.
    while viewer.is_running():
        step_start = time.time()
        
        # Only update servo commands at 10 Hz (every 0.1 seconds)
        if data.time - last_servo_update >= servo_update_period:
            # Calculate desired positions using sinusoidal trajectories
            # Position control: data.ctrl now represents desired joint angles
            data.ctrl[0] = gait_params[0][0] + (gait_params[0][1] * math.sin((frequency * data.time) + gait_params[0][2]))
            data.ctrl[1] = gait_params[1][0] + (gait_params[1][1] * math.sin((frequency * data.time) + gait_params[1][2]))
            data.ctrl[2] = gait_params[2][0] + (gait_params[2][1] * math.sin((frequency * data.time) + gait_params[2][2]))
            data.ctrl[3] = gait_params[3][0] + (gait_params[3][1] * math.sin((frequency * data.time) + gait_params[3][2]))
            
            # build observation like training env: roll,pitch,yaw and current 4 joint qpos
            quat = data.qpos[3:7]
            euler = quat_to_euler(quat)
            joint_pos = data.qpos[7:11].astype(np.float32)
            obs = np.concatenate([euler.astype(np.float32), joint_pos]).astype(np.float32)
            
            last_servo_update = data.time
        
        # Logging (only log joint angles, not freejoint DOFs)
        time_log.append(data.time)
        for j in range(num_joints):
            q_log[j].append(data.qpos[7+j])  # Skip first 7 DOFs (freejoint)
            qd_log[j].append(data.qvel[6+j])  # Skip first 6 DOFs (freejoint velocities)
            q_desired_log[j].append(data.ctrl[j])
            actuator_force_log[j].append(data.actuator_force[j])

        # Step the simulation
        mujoco.mj_step(model, data)

        # Sync viewer
        viewer.sync()

        # Rudimentary time keeping: sleep for the remainder of the timestep
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        
# -------- Plot after simulation --------
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
plt.savefig("motor_plots_position.png")
print("Saved motor_plots_position.png")
