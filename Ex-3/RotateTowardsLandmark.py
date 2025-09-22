from time import perf_counter, sleep
from rightSpeedModifier import rightSpeedModifier
import sys

import numpy as np

from robot_extended import RobotExtended

arlo_master = RobotExtended()
arlo = arlo_master.robot

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def go_to_landmark(target_landmark: int, total_time=10):
    start = perf_counter()
    isGoing = True
    while (isGoing):
        if (perf_counter() - start > total_time):
            print(arlo.stop())
            isGoing = False

        leftSpeed = 43
        rightSpeed = leftSpeed + rightSpeedModifier[leftSpeed]

        table = arlo_master.perform_image_analysis_table()
        if target_landmark in table and np.abs(table[target_landmark].rvecs[2]) > 0.1:
            withclock = table[target_landmark].rvecs[2] > 0
            if withclock:
                print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
            else:
                print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
            sleep(0.1)
        else:
            print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
            sleep(0.1)


go_to_landmark(2)

        
            

    
