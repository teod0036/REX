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
    while (True):
        landmarks = arlo_master.perform_image_analysis_table()
        
        if target_landmark not in landmarks: 
            perform_Turn90(True, 0.1)
            print(arlo.stop())
            continue

        if landmarks[target_landmark].tvecs[0] < -0.05:
            perform_Turn90(False, 0.1)
            print(arlo.go_diff(64, 64 + rightSpeedModifier[64], 1, 1))
            print(arlo.stop())
        elif landmarks[target_landmark].tvecs[0] > 0.05:
            perform_Turn90(True, 0.1)
            print(arlo.go_diff(64, 64 + rightSpeedModifier[64], 1, 1))
            print(arlo.stop())
        else:
            if landmarks[target_landmark].tvecs[2] < 0.1:
                print(arlo.stop())
                return
            print(arlo.go_diff(64, 64 + rightSpeedModifier[64], 1, 1))

go_to_landmark(2)

        
            

    