from time import perf_counter, sleep

import robot

is_low_on_battery = True
arlo = robot.Robot()

def turn(params):
    withclock, degrees = params

    # Speed constants
    leftSpeed = 64
    rightSpeed = 64
    rightSpeedmodifier = 1

    turnsleep = 0.347
    out_of_battery = 0.347
    extraconst_c = -0.005
    extraconst_nc = 0.005
    if is_low_on_battery:
        turnsleep += out_of_battery

    if withclock:
        print(arlo.go_diff(leftSpeed, rightSpeed + rightSpeedmodifier, 1, 0))
        sleep((turnsleep + extraconst_c) * (degrees / 45))
    else:
        print(arlo.go_diff(leftSpeed, rightSpeed + rightSpeedmodifier, 0, 1))
        sleep((turnsleep + extraconst_nc) * (degrees / 45))

    print(arlo.stop())
    sleep(0.1)

    return 0

def forward(distance):
    # Time constants
    # note: tuning for 0.5 meters, so distance is scaled by 2

    c = 1.15 + 0.0125
    go_sleep = c * distance * 2
    out_of_battery = 1.15
    if is_low_on_battery:
        go_sleep += out_of_battery
    '''
    go_sleep = c * distance * 2
    distance = 1/2 * go_sleep / c
    '''

    leftSpeed = 64
    rightSpeed = 64
    rightSpeedmodifier = 1

    print(arlo.go_diff(leftSpeed, rightSpeed + rightSpeedmodifier, 1, 1))

    # roughly equivalent to sleep(0.1) but with sensor detection
    start = perf_counter()
    isgoing = True

    while isgoing:
        if perf_counter() - start > go_sleep:
            print(arlo.stop())
            isgoing = False
        
        front_dist = arlo.read_front_ping_sensor()
        left_dist = arlo.read_left_ping_sensor()
        right_dist = arlo.read_right_ping_sensor()
        if ((front_dist < 100 and front_dist != -1) or
            (left_dist < 100 and left_dist != -1) or
            (right_dist < 100 and right_dist != -1)):
            print(arlo.stop())

            end = perf_counter() - start 
            distance_driven = end / (2 * c)
            
            if left_dist < 300 and left_dist != -1:
                distance_driven = distance_driven * -1
            
            sleep(0.1)
            return distance_driven
    
    sleep(0.1)
    print(arlo.stop())

    return 0




if __name__ == "__main__":
    print(f"distance driven: {forward(2)}");
