from time import perf_counter
import robot
from Turn90 import perform_Turn90

arlo = robot.Robot()

def sees_front(distance=500):
    return arlo.read_front_ping_sensor() < distance and arlo.read_front_ping_sensor() > -1

def sees_left(distance=300):
    return arlo.read_left_ping_sensor() < distance and arlo.read_front_ping_sensor() > -1

def sees_right(distance=300):
    return arlo.read_right_ping_sensor() < distance and arlo.read_front_ping_sensor() > -1

def go_and_avoid(go_time=10):

    print(arlo.go_diff(64, 64, 1, 1))
    start = perf_counter()
    isGoing = True
    while (isGoing): # or some other form of loop
        if (perf_counter() - start > go_time): # Stop after 5 seconds
            print(arlo.stop())
            isGoing = False

        #If front sensor doesn't see anything and  
        # left and right sensors are further than 10 cm from something continue forwards
        #This code is to support going through tunnels 
        if (not sees_right(100) and not sees_left(100) and not sees_front()):
            continue
        
        #If front sensor detects something turn 90 degrees right
        if (sees_front()):
            print(perform_Turn90(True))
            print(arlo.go_diff(64, 64, 1, 1))

        #If left sensor detects something turn 45 degrees right
        if (sees_left()):
            print(perform_Turn90(True, 0.347))
            print(arlo.go_diff(64, 64, 1, 1))

        #If right sensor detects something turn 45 degrees left
        if (sees_right()):
            print(perform_Turn90(False, 0.347))
            print(arlo.go_diff(64, 64, 1, 1))

go_and_avoid()