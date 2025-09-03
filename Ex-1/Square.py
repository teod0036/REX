from time import sleep
import robot
from oneMeter import go_meter
from Turn90 import perform_Turn90

arlo = robot.Robot()

def perform_square():
    for i in range(4):
        #go straight
        go_meter()

        #turn right
        perform_Turn90(withclock=False)

perform_square()