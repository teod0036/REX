from time import sleep
import robot

arlo = robot.Robot()

def turn(params):
    withclock, degrees = params
    # Speed constants
    leftSpeed = 64  
    rightSpeed = 64 - 3
    if withclock:
        print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
    else:
        print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))

    sleep(0.694 * degrees/90)

def forward(distance):
    # Time constants
    go_sleep = 2.3 * distance

    # Speed constants
    leftSpeed = 64
    rightSpeed = 64
    rightSpeedmodifier = -3

    #go straight
    print(arlo.go_diff(leftSpeed, rightSpeed + rightSpeedmodifier, 1, 1))
    sleep(go_sleep)