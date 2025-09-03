from time import sleep
import robot

arlo = robot.Robot()

def perform_Turn90():

    # Speed constants
    leftSpeed = 64  
    rightSpeed = 64

    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
    sleep(0.7)



perform_Turn90()
perform_Turn90()
perform_Turn90()
perform_Turn90()
perform_Turn90()
perform_Turn90()
perform_Turn90()
perform_Turn90()


