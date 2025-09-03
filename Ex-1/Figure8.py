from time import sleep
import robot
from oneMeter import go_meter
arlo = robot.Robot()

def PerformFigureEight():
    # Go Straight
    leftSpeed = 64
    rightSpeed = 64
    Rightspeedmodifier = -8
    turntimer = 3.2
    go_meter()

    # Start Turn
    print(arlo.stop())
    print(arlo.go_diff(41, rightSpeed + 50 + Rightspeedmodifier, 1, 1))

    sleep(turntimer)
    # Go Straight

    go_meter()
    # Start Turn
    print(arlo.go_diff(leftSpeed + 50, 41, 1, 1))

    sleep(turntimer)
    # Should be at start position now
    print(arlo.stop())


PerformFigureEight()
