from time import sleep, perf_counter
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

    return 0


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
    start = perf_counter()
    isgoing = True
    print(arlo.go_diff(leftSpeed, rightSpeed + rightSpeedmodifier, 1, 1))
    while isgoing:
        if perf_counter() - start > go_sleep:
            print(arlo.stop())
            isgoing = False
        
        front_dist = arlo.read_front_ping_sensor()
        if front_dist < 200 and front_dist != -1:
            end = start - perf_counter()
            print(arlo.stop())
            distance_driven = end * (10/(10 * 2.3) + (10 * extraconst)) 
            
            right_dist = arlo.read_right_ping_sensor()
            if right_dist < 100 and right_dist != -1:
                distance_driven = distance_driven * -1
            
            sleep(0.1)
            return distance_driven
    
    sleep(0.1)

    return 0

if __name__ == "__main__":
    # for i in range(4):
    turn((False, 30))
    print(arlo.stop())
    sleep(0.1)
    turn((False, 30))
    print(arlo.stop())
    sleep(0.1)
    turn((False, 30))
    print(arlo.stop())
