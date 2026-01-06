from pylx16a.lx16a import *
import sys
import termios
import tty
import select
import time
import numpy as np
from stable_baselines3 import PPO
import math
import threading

sys.path.append('../9DoF_MARG_Madgwick_Filter/Raspberry_pi/Madwick_filter')
import Madgwick_Filter as mf

LX16A.initialize("/dev/ttyUSB0", 0.1)

# take in list of tuples of servo IDs and the positions you want to move them to, moves them all smoothly at the same time
def smooth_move(moves):
    '''
    input moves: [(servo ID, angle),...]
    '''
    path_divisions = 15
    paths = []
    print(moves)
    for (ID, endpos) in moves:
        start = LX16A(ID).get_physical_angle()
        path = np.linspace(start, endpos, num = path_divisions)
        paths.append((ID, path))
    for i in range(path_divisions):
        for (ID, path) in paths:
            LX16A(ID).move(path[i])

def transition(pos):
    LX16A(3).move(90)
    LX16A(4).move(90)

    if pos == "startup":
        smooth_move([(2, 100), (5, 100)])
    elif pos == "homing":
        smooth_move([(2, 45), (5, 45)])
        
# --- Setup non-blocking keyboard input ---
def get_key():
    """Return a pressed key character, or None if no key was pressed."""
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None

def PPO_move():
    '''
    Docstring for PPO_move
    servo ID -> sim label:
    4 -> left_shoulder
    5 -> left_elbow
    3 -> right_shoulder
    2 -> right elbow
    '''
    model = PPO.load("ppo_gait_actuator.zip")
    euler = np.array(list(mf.latest_orientation.values()))
    joint_pos = np.array([LX16A(4).get_physical_angle(), LX16A(5).get_physical_angle(), LX16A(3).get_physical_angle(), LX16A(2).get_physical_angle()])
    obs = np.concatenate([euler.astype(np.float32), joint_pos.astype(np.float32)]).astype(np.float32)
    
    # convert inputs
    obs[0] = -(obs[0] * (math.pi/180))
    obs[1] = -(obs[1] * (math.pi/180))
    obs[2] = obs[2] * (math.pi/180)
    obs[3] = -((obs[3] - 90) * (math.pi/180))
    obs[4] = -((obs[4] - 100) * (math.pi/180))
    obs[5] = (obs[5] - 90) * (math.pi/180)
    obs[6] = (obs[6] - 100) * (math.pi/180)
    action, _ = model.predict(obs, deterministic=True)
    
    #convert outputs
    action = action.astype(np.float64)
    action[0] = 90 - (action[0] * (180/math.pi))
    action[1] = 100 - (action[1] * (180/math.pi))
    action[2] = (action[2] * (180/math.pi)) + 90
    action[3] = (action[3] * (180/math.pi)) + 100

    # apply action to servos (action interpreted as absolute desired positions)
    try:
        LX16A(4).move(action[0])
        LX16A(5).move(action[1])
        LX16A(3).move(action[2])
        LX16A(2).move(action[3])
    except ServoTimeoutError as e:
        print(f"Servo {e.id_} is not responding. Exiting...")
        quit()

def main():
    num_servos = 6
    try:
        for i in range(2, num_servos):
            LX16A(i).set_angle_limits(0, 240)
    except ServoTimeoutError as e:
        print(f"Servo {e.id_} is not responding. Exiting...")
        quit()

    neutral_angles = transition("startup")

    filter_thread = threading.Thread(target=mf.main, daemon=True)
    filter_thread.start()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    print("Press W to move (RL), Enter to stop, Ctrl+C to quit.")

    try:
        while True:
            time.sleep(0.1)  # Adjust loop rate as needed
            PPO_move()

    finally:
        transition("homing")
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\nExiting cleanly.")

if __name__ == "__main__":
    main()
