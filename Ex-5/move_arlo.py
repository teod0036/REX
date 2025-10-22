from time import sleep
import robot

arlo = robot.Robot()

def turn(params):
    withclock, degrees = params
    # Speed constants
    out_of_battery = 0.694
    leftSpeed = 64  
    rightSpeed = 64 - 1
    extraconst_c = 0
    extraconst_nc = 0
    if withclock:
        print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
        sleep((0.694 + extraconst_c) * (degrees/90))
    else:
        print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
        sleep((0.694 + extraconst_nc) * (degrees/90))
    print(arlo.stop())
    sleep(0.1)


def forward(distance):
    # Time constants
    out_of_battery = 2.3
    extraconst = 0.025
    go_sleep = (2.3 + extraconst) * distance

    # Speed constants
    leftSpeed = 64
    rightSpeed = 64
    rightSpeedmodifier = -1

    #go straight
    print(arlo.go_diff(leftSpeed, rightSpeed + rightSpeedmodifier, 1, 1))
    sleep(go_sleep)
    print(arlo.stop())
    sleep(0.1)

if __name__ == "__main__":
    # for i in range(4):
    turn((True, 45))
