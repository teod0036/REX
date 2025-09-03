from time import sleep
import robot

arlo = robot.Robot()

def TestWheels(withclock: bool, sleepduration: float = 0.5):
    

    # Speed constants
    leftSpeed = 64  
    rightSpeed = 64
    Rightspeedmodifier = -1
    if withclock:
        print(arlo.go_diff(0, rightSpeed + Rightspeedmodifier, 1, 1))
    else:
        print(arlo.go_diff(leftSpeed, 0, 1, 1))

    sleep(sleepduration)



TestWheels(True)
TestWheels(True)
TestWheels(True)
TestWheels(True)
TestWheels(True)
TestWheels(True)
TestWheels(True)
TestWheels(True)






