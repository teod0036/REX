from time import sleep
import robot

arlo = robot.Robot()

def go_meter():
    # Time constants
    go_sleep = 2

    # Speed constants
    leftSpeed = 50
    rightSpeed = 50

    #go straight
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
    sleep(go_sleep)    

go_meter()