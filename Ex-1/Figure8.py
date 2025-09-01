from time import sleep
import robot

arlo = robot.Robot()

def PerformFigureEight():
    # Go Straight
    leftSpeed = 64
    rightSpeed = 64
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))

    sleep(1)
    # Start Turn
    print(arlo.stop())
    print(arlo.go_diff(leftSpeed / 2, rightSpeed, 1, 1))

    sleep(1)
    # Go Straight
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))

    sleep(1)
    # Start Turn
    print(arlo.stop())
    print(arlo.go_diff(leftSpeed, rightSpeed / 2, 1, 1))

    sleep(1)
    # Should be at start position now
    print(arlo.stop())

PerformFigureEight()
    