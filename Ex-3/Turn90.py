from time import sleep
import robot

arlo = robot.Robot()

def perform_Turn90(withclock: bool, sleepduration: float = 0.694):
    # Speed constants
    leftSpeed = 64  
    rightSpeed = 64 - 3
    if withclock:
        print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
    else:
        print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))

    sleep(sleepduration)
