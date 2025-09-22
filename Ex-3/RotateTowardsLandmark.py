from time import perf_counter, sleep
from rightSpeedModifier import rightSpeedModifier
import sys

import numpy as np

from robot_extended import RobotExtended

arlo_master = RobotExtended()
arlo = arlo_master.robot

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def go_to_landmark(target_landmark: int):
    # start = perf_counter()
    isGoing = True
    while (isGoing):
        # if (perf_counter() - start > total_time):
        #     eprint(arlo.stop())
        #     isGoing = False

        leftSpeed = 43
        rightSpeed = leftSpeed + rightSpeedModifier[leftSpeed]

        sleep(1)
        table = arlo_master.perform_image_analysis_table()
        eprint(table)

        if target_landmark in table:
            # if (np.abs(table[target_landmark].rvecs[2]) > 0.01):
            withclock = table[target_landmark].rvecs[2] < 0
            if withclock:
                eprint(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
            else:
                eprint(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
            sleep(np.max(0.05, np.abs(table[target_landmark].rvecs[2])))
            eprint(arlo.stop())
            # else:
            #     eprint(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
            #     sleep(0.2)
            #     eprint(arlo.stop())
        else:
            arlo.go_diff(leftSpeed, rightSpeed, 1, 0)
            sleep(0.1)
            arlo.stop()


go_to_landmark(2)

        
            

    
