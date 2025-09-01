from time import sleep
import robot

arlo = robot.Robot()

def PerformFigureEight():
    # Go Straight
    leftSpeed = 64
    rightSpeed = 64
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))

    sleep(2)
    # Start Turn
    print(arlo.stop())
    print(arlo.go_diff(41, 127, 1, 1))

    sleep(2.5)
    # Go Straight
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))

    sleep(1.7)
    # Start Turn
    print(arlo.go_diff(127, 41, 1, 1))

    sleep(3)
    # Should be at start position now
    print(arlo.stop())

PerformFigureEight()
PerformFigureEight()
PerformFigureEight()
