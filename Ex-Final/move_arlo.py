from time import sleep, perf_counter
import robot

arlo = robot.Robot()

# def turn(params):
#     withclock, degrees = params
#     # Speed constants
#     out_of_battery = 0.694
#     leftSpeed = 64  
#     rightSpeed = 64 - 1
#     extraconst_c = 0
#     extraconst_nc = 0
#     if withclock:
#         print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
#         sleep((0.694 + extraconst_c) * (degrees/90))
#     else:
#         print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
#         sleep((0.694 + extraconst_nc) * (degrees/90))
#     print(arlo.stop())
#     sleep(0.1)
#
#     return 0

def forward(distance):
    # Time constants
    # note: tuning for 0.5 meters, so distance is scaled by 2
    extraconst = 0.0125
    go_sleep = (1.15 + extraconst) * distance * 2

    leftSpeed = 64
    rightSpeed = 64
    rightSpeedmodifier = +2

    print(arlo.go_diff(leftSpeed, rightSpeed + rightSpeedmodifier, 1, 1))
    sleep(go_sleep)

if __name__ == "__main__":
    forward(0.5)
