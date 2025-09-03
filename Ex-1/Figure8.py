from time import sleep
import robot
from oneMeter import go_meter
arlo = robot.Robot()

def PerformFigureEight():
    # Go Straight
    leftSpeed = 64
    rightSpeed = 64
    Rightspeedmodifier = -2
    go_meter()

    # Start Turn
    print(arlo.stop())
    print(arlo.go_diff(43, 127 + Rightspeedmodifier, 1, 1))

    sleep(2.5)
    # Go Straight

    go_meter()
    # Start Turn
    print(arlo.go_diff(127, 43 + Rightspeedmodifier, 1, 1))

    sleep(2.5)
    # Should be at start position now
    print(arlo.stop())


PerformFigureEight()
