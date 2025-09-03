from time import sleep
import robot

arlo = robot.Robot()

def PerformTurn90():
    # Go Straight
    leftSpeed = 64
    rightSpeed = 64
    print(arlo.stop())
    print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
    sleep(1)
    print(arlo.stop())

PerformTurn90()
