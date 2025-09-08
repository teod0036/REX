from time import perf_counter, sleep

import robot
from rightSpeedModifier import rightSpeedmodifier

arlo = robot.Robot()

def perform_Turn(withclock: bool, angle: float):
    turn90_duration = 0.694 * angle / 90

    leftSpeed = 64  
    rightSpeed = 64 + rightSpeedmodifier[leftSpeed]
    if withclock:
        print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
    else:
        print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))

    sleep(turn90_duration)

def main(go_time=10):
    start = perf_counter()
    isGoing = True
    while isGoing:  # or some other form of loop
        if perf_counter() - start > go_time:  # Stop after 5 seconds
            print(arlo.stop())
            isGoing = False
        print(arlo.read_front_ping_sensor())
        sleep(1)


# if __name__ == "__main__":
#     main()


