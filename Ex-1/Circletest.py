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
    print(arlo.go_diff(0, 62 + Rightspeedmodifier, 1, 1))
    sleep(turntimer)

def PerformTurnRightCircle():
    leftSpeed = 64
    rightSpeed = 64
    turntimer = 3.2
    print(arlo.go_diff(64, 0, 1, 1))
    sleep(turntimer)


PerformTurnLeftCircle()
PerformTurnLeftCircle()


