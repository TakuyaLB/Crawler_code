from pylx16a.lx16a import *
import sys
import termios
import tty
import select
import time
import threading
import numpy as np

sys.path.append('../9DoF_MARG_Madgwick_Filter/Raspberry_pi/Madwick_filter')
import Madgwick_Filter as mf

import PID_controller as pidc

LX16A.initialize("/dev/ttyUSB0", 0.1)

num_servos = 6

# Servo neutral position
SERVO_NEUTRAL = 90

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

    start_2 = LX16A(2).get_physical_angle()
    start_5 = LX16A(5).get_physical_angle()

    if pos == "startup":
        smooth_move([(2, 100), (5, 100)])
    elif pos == "homing":
        smooth_move([(2, 45), (5, 45)])
    neutral_angles = {"2": LX16A(2).get_physical_angle(), "3": LX16A(3).get_physical_angle(), "4": LX16A(4).get_physical_angle(), "5": LX16A(5).get_physical_angle()}
    return neutral_angles

def balance(pid, neutral_angles):
    '''
    # Get current pitch from your sensor
    pitch = mf.latest_orientation['roll']
        
    # Setpoint is 0 (level)
    setpoint = 0
        
    # Get PID correction
    correction = pid.update(setpoint, pitch)
        
    # Apply correction to servo
    servo_3_position = SERVO_NEUTRAL + correction
    servo_4_position = SERVO_NEUTRAL - correction

    LX16A(3).move(servo_3_position)
    LX16A(4).move(servo_4_position)
    '''

    # Clamp to valid servo range
    #servo_position = max(0, min(180, servo_position))
        
    # Send to servo
    #set_servo_position(servo_position)  # Replace with your servo control

    #orientation = mf.latest_orientation
    #print(orientation)
    #adjustments = [(3, int(neutral_angles['3']-orientation['roll'])), (4, int(neutral_angles['4']-orientation['roll'])), (2, int(neutral_angles['2']-orientation['pitch'])), (5, int(neutral_angles['5']-orientation['pitch']))]
    adjustments = [(3, 90), (4, 90), (2, 100), (5, 100)]
    smooth_move(adjustments)
    '''
    LX16A(3).move(neutral_angles['3']-orientation['roll'])
    LX16A(4).move(neutral_angles['4']+orientation['roll'])

    LX16A(2).move(neutral_angles['2']-orientation['pitch'])
    LX16A(5).move(neutral_angles['5']+orientation['pitch'])
    '''
    print(mf.latest_orientation)

def basic_move(dir):
    try:
        if dir == "forward":
            LX16A(1).motor_mode(-1000)
            LX16A(6).motor_mode(1000)
        elif dir == "backwards":
            LX16A(1).motor_mode(1000)
            LX16A(6).motor_mode(-1000)
        elif dir == "left":
            LX16A(1).motor_mode(-1000)
            LX16A(6).motor_mode(0)
        elif dir == "right":
            LX16A(1).motor_mode(0)
            LX16A(6).motor_mode(1000)
        elif dir == "spin left":
            LX16A(1).motor_mode(-1000)
            LX16A(6).motor_mode(-1000)
        elif dir == "spin right":
            LX16A(1).motor_mode(1000)
            LX16A(6).motor_mode(1000)
    except ServoTimeoutError as e:
        print(f"Servo {e.id_} is not responding. Exiting...")
        quit()

# --- Setup non-blocking keyboard input ---
def get_key():
    """Return a pressed key character, or None if no key was pressed."""
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None

def main():
    pid = pidc.PIDController(
        kp=1.0,      # Start here and adjust
        ki=0.0,      # Low to prevent windup
        kd=0.0,      # Helps reduce oscillation
        output_limits=(-45, 45)  # Limit correction range
    )

    filter_thread = threading.Thread(target=mf.main, daemon=True)
    filter_thread.start()

    print("starting Madgwick filter...")
    time.sleep(2)

    try:
        for i in range(2, num_servos):
            LX16A(i).set_angle_limits(0, 240)
    except ServoTimeoutError as e:
        print(f"Servo {e.id_} is not responding. Exiting...")
        quit()

    neutral_angles = transition("startup")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    print("Press W/A/S/D to move, Z/X to spin, Ctrl+C to quit.")
    #pressed_key = None

    try:
        while True:
            key = get_key()

            # Handle new key press
            if key:
                if key == '\x03':  # Ctrl+C to quit
                    break
                elif key == 'w':
                    print("forward")
                    basic_move("forward")
                elif key == 's':
                    print("backwards")
                    basic_move("backwards")
                elif key == 'a':
                    print("left")
                    basic_move("left")
                elif key == 'd':
                    print("right")
                    basic_move("right")
                elif key == 'z':
                    print("spin left")
                    basic_move("spin left")
                elif key == 'x':
                    print("spin right")
                    basic_move("spin right")
                elif key == '\n':  # Enter key stops
                    try:
                        LX16A(1).servo_mode()
                        LX16A(6).servo_mode()
                    except ServoTimeoutError as e:
                        print(f"Servo {e.id_} is not responding. Exiting...")
                        quit()
                    print("Stopped.")

            balance(pid, neutral_angles) # Keep adjusting the arms to keep main body flat

            time.sleep(0.1)  # Adjust loop rate as needed

    finally:
        transition("homing")
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\nExiting cleanly.")

if __name__ == "__main__":
    main()
