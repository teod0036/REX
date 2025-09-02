from time import sleep
import robot
from oneMeter import go_meter

arlo = robot.Robot()

def perform_square():
    # Time constants
    turn_sleep = 2

    # Speed constants
    leftSpeed = 50
    rightSpeed = 50

    for i in range(4):
        #go straight
        go_meter()

        #turn right
        print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
        sleep(turn_sleep)


perform_square()