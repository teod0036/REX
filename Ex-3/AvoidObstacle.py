from time import perf_counter
import robot
from Turn90 import perform_Turn90

arlo = robot.Robot()

rmod = -3

def sees_front(distance=500):
    return arlo.read_front_ping_sensor() < distance and arlo.read_front_ping_sensor() > -1

def sees_left(distance=300):
    return arlo.read_left_ping_sensor() < distance and arlo.read_left_ping_sensor() > -1

def sees_right(distance=300):
    return arlo.read_right_ping_sensor() < distance and arlo.read_right_ping_sensor() > -1

def sees_behind(distance=300):
    return arlo.read_back_ping_sensor() < distance and arlo.read_back_ping_sensor() > -1

def sees_all():
    return sees_front() and sees_behind() and sees_left() and sees_right()

def in_tunnel():
    return sees_right() and sees_left() and not sees_front()

def go_and_avoid(go_time=10, tunnelwidth=100):

    print(arlo.go_diff(64, 64 + rmod, 1, 1))
    start = perf_counter()
    isGoing = True
    while (isGoing): # or some other form of loop
        if (perf_counter() - start > go_time): # Stop after 5 seconds
            print(arlo.stop())
            isGoing = False

        #If front sensor doesn't see anything and left and right sensors see something the robot is in a tunnel
        #This code is to support going through tunnels 
        if (in_tunnel()):
            #If the tunnel has narrowed turn around and go back
            if (sees_left(tunnelwidth) and sees_right(tunnelwidth)):
                perform_Turn90(True)
                perform_Turn90(True)
                print(arlo.go_diff(52, 52 + rmod, 1, 1))
                continue
            #If robot is too close to left tunnel wall turn a little bit right
            if (sees_left(tunnelwidth)):
                perform_Turn90(True, 0.1)
                print(arlo.go_diff(52, 52 + rmod, 1, 1))
                continue 
            #If robot is too close to right tunnel wall turn a litlle bit left
            if (sees_right(tunnelwidth)):
                perform_Turn90(False, 0.1)
                print(arlo.go_diff(52, 52 + rmod, 1, 1))
                continue

            print(arlo.go_diff(52, 52 + rmod, 1, 1))
            continue
        
        #If front sensor detects something turn 90 degrees right
        if (sees_front()):
            perform_Turn90(True)
            print(arlo.go_diff(64, 64  + rmod, 1, 1))

        #If left sensor detects something turn 45 degrees right
        if (sees_left()):
            perform_Turn90(True, 0.347)
            print(arlo.go_diff(64, 64 + rmod, 1, 1))

        #If right sensor detects something turn 45 degrees left
        if (sees_right()):
            perform_Turn90(False, 0.347)
            print(arlo.go_diff(64, 64  + rmod, 1, 1))

        #If the robot is surrounded it spins a bit and checks again (It could have been kidnapped)
        if (sees_all()):
            perform_Turn90(True, 0.1)
            print(arlo.stop())

go_and_avoid()
print(arlo.stop())