from time import sleep
import robot

arlo = robot.Robot()

def turn(params):
    withclock, degrees = params
    # Speed constants
    leftSpeed = 64  
    rightSpeed = 64 - 3
    extraconst_c = 0.2
    extraconst_nc = 0.25
    if withclock:
        print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
        sleep((0.694 + extraconst_c) * degrees/90)
    else:
        print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
        sleep((0.694 + extraconst_nc) * degrees/90)


def forward(distance):
    # Time constants
    extraconst = 0.025
    go_sleep = (2.3 + extraconst) * distance

    # Speed constants
    leftSpeed = 64
    rightSpeed = 64
    rightSpeedmodifier = -1

    #go straight
    print(arlo.go_diff(leftSpeed, rightSpeed + rightSpeedmodifier, 1, 1))
    sleep(go_sleep)

if __name__ == "__main__":
    forward(1)