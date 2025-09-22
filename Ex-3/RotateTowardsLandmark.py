from time import perf_counter, sleep
from rightSpeedModifier import rightSpeedModifier
import sys

import numpy as np
import numpy.linalg as linalg


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
            # if (np.abs(table[target_landmark].rvec[2]) > 0.1):
            #     withclock = table[target_landmark].rvec[2] < 0
            #     if withclock:
            #         print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
            #     else:
            #         print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
            #     sleep(0.1)
            #     print(arlo.stop())
            # else:
            if (linalg.norm(table[target_landmark].tvec)) >= 1:
                print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
                sleep(0.5)
                print(arlo.stop())
            else:
                print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
                sleep(1.5)
                print(arlo.stop())
        else:
            arlo.go_diff(leftSpeed, rightSpeed, 1, 0)
            sleep(0.5)
            arlo.stop()


go_to_landmark(2)

        
            

    
