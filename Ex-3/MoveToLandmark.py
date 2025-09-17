from time import perf_counter
import robot
from Turn90 import perform_Turn90
from FindLandmark import perform_Findlandmark
from rightSpeedModifier import rightSpeedModifier

arlo = robot.Robot()

def go_to_landmark(target_landmark):
    while (True):
        maybe_landmark = perform_Findlandmark()
        if maybe_landmark is not None:
            tvecs = maybe_landmark
        else:
            perform_Turn90(True, 0.1735)

        if tvecs[target_landmark][0] < -0.0001:
            perform_Turn90(True, 0)
        elif tvecs[target_landmark][0] > 0.0001:
            perform_Turn90(False, 0)
        else:
            if tvecs[target_landmark][2] < 0.1:
                print(arlo.stop())
                return
            print(arlo.go_diff(64, 64 + rightSpeedModifier[64], 1, 1))

go_to_landmark(2)

        
            

    