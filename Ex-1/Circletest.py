from time import sleep
import robot
from oneMeter import go_meter
arlo = robot.Robot()

def PerformTurnLeftCircle():
    # Go Straight
    leftSpeed = 64
    rightSpeed = 64
    Rightspeedmodifier = -2
    turntimer = 3.2
    print(arlo.go_diff(0, 114, 1, 1))
    sleep(turntimer)

def PerformTurnRightCircle():
    leftSpeed = 64
    rightSpeed = 64
    Rightspeedmodifier = -2
    turntimer = 3.2
    print(arlo.go_diff(114, 0, 1, 1))
    sleep(turntimer)


PerformTurnLeftCircle()
PerformTurnLeftCircle()


