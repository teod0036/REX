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
    print(arlo.go_diff(10, rightSpeed, 1, 1))

    sleep(3.5)
    # Go Straight
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))

    sleep(2)
    # Start Turn
    print(arlo.go_diff(leftSpeed, 10, 1, 1))

    sleep(3.5)
    # Should be at start position now
    print(arlo.stop())

PerformFigureEight()
PerformFigureEight()
PerformFigureEight()
PerformFigureEight()
PerformFigureEight()
