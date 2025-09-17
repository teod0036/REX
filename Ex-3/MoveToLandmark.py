from time import perf_counter
import robot
from Turn90 import perform_Turn90
from FindLandmark import perform_Findlandmark

arlo = robot.Robot()

def go_to_landmark(target_landmark):
    while (True):
        maybe_landmark = perform_Findlandmark()
        if maybe_landmark is not None:
            rvec, tvec = maybe_landmark
        else:
            perform_Turn90(True, 0.1735)

        if tvec[0] < -0.00001:
            perform_Turn90(True, 0)
        elif tvec[0] > 0.00001:
            perform_Turn90(False, 0)
        else:
            if tvec[2] < 0.001:
                print(arlo.stop())
                return
            print(arlo.go_diff(64, 64, 1, 1))


        
            

    