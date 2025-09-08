from time import perf_counter
import robot
from Turn90 import perform_Turn90

arlo = robot.Robot()

def go_and_avoid(go_time=5):

    print(arlo.go_diff(64, 64, 1, 1))
    start = perf_counter()
    isGoing = True
    while (isGoing): # or some other form of loop
        if (perf_counter() - start > go_time): # Stop after 5 seconds
            print(arlo.stop())
            isGoing = False
        if (arlo.read_front_ping_sensor() < 150 and arlo.read_front_ping_sensor() > -1):
            print(perform_Turn90(True))
            print(arlo.go_diff(64, 64, 1, 1))

go_and_avoid()