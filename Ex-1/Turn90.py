from time import sleep
import robot

arlo = robot.Robot()

def perform_Turn90():

    # Speed constants
    leftSpeed = 64  
    rightSpeed = 64

    print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
    sleep(1)



perform_Turn90()