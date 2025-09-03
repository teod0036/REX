from time import sleep
import robot

arlo = robot.Robot()

def perform_Turn90(withclock: bool, sleepduration: float = 0.685):
    

    # Speed constants
    leftSpeed = 64  
    rightSpeed = 64
    if withclock:
        print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
    else:
        print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))

    sleep(sleepduration)



perform_Turn90(False)
perform_Turn90(False)
perform_Turn90(False)
perform_Turn90(False)
perform_Turn90(False)
perform_Turn90(False)
perform_Turn90(False)
perform_Turn90(False)



