from time import perf_counter, sleep

import robot_extended

from Turn90 import perform_Turn90
from rightSpeedModifier import rightSpeedModifier
import sys

arlo_master = robot_extended.RobotExtended()
arlo = arlo_master.robot

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def go_to_landmark(target_landmark):
    last_turn_direction = True
    while (True):
        landmarks = arlo_master.perform_image_analysis_table()
        
        if target_landmark not in landmarks: 
            perform_Turn90(last_turn_direction, 0.15)
            print(arlo.stop())
            continue

        if landmarks[target_landmark].tvec[0] < -0.05:
            last_turn_direction = True
            perform_Turn90(False, 0.05)
            print(arlo.go_diff(64, 64 + rightSpeedModifier[64], 1, 1))
        elif landmarks[target_landmark].tvec[0] > 0.05:
            last_turn_direction = False
            perform_Turn90(True, 0.05)
            print(arlo.go_diff(64, 64 + rightSpeedModifier[64], 1, 1))
        else:
            if landmarks[target_landmark].tvec[2] < 0.2 or arlo.read_front_ping_sensor() < 20:
                print(arlo.stop())
                eprint(f"I have arrived at landmark {target_landmark}")
                return
            print(arlo.go_diff(64, 64 + rightSpeedModifier[64], 1, 1))

go_to_landmark(2)

        
            

    