from time import perf_counter, sleep
import robot
from Turn90 import perform_Turn90
from imagecapture import initCamera
from FindLandmark import perform_Findlandmark
from rightSpeedModifier import rightSpeedModifier
import sys

arlo = robot.Robot()

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def go_to_landmark(target_landmark):
    cam = initCamera()
    while (True):
        maybe_landmark = perform_Findlandmark(cam)
        if maybe_landmark is not None:
            tvecs = maybe_landmark
        else:
            perform_Turn90(True, 0.1)
            print(arlo.stop())
            continue

        eprint(f"left/right offset: {tvecs[target_landmark][1][0][0]}")
        eprint(f"Distance to landmark in meters: {tvecs[target_landmark][1][0][2]}")

        if tvecs[target_landmark][1][0][0] < -0.01:
            perform_Turn90(True, 0)
        elif tvecs[target_landmark][1][0][0] > 0.01:
            perform_Turn90(False, 0)
        else:
            if tvecs[target_landmark][1][0][2] < 0.1:
                print(arlo.stop())
                cam.close()
                return
            print(arlo.go_diff(64, 64 + rightSpeedModifier[64], 1, 1))

go_to_landmark(2)

        
            

    