from time import sleep
import robot

arlo = robot.Robot()

def go_meter():
    # Time constants
    go_sleep = 2

    # Speed constants
    leftSpeed = 64
    rightSpeed = 64
    rightSpeedmodifier = -2

    #go straight
    print(arlo.go_diff(leftSpeed, rightSpeed + rightSpeedmodifier, 1, 1))
    sleep(go_sleep)    

go_meter()