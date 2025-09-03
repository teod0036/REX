from time import sleep
import robot

arlo = robot.Robot()

def TestWheels(userightwheel: bool, sleepduration: float = 0.5):
        # Speed constants
    speed = 41
    Rightspeedmodifier = 0
    if userightwheel:
        print(arlo.go_diff(0, speed + Rightspeedmodifier, 1, 1))
    else:
        print(arlo.go_diff(speed, 0, 1, 1))

    sleep(sleepduration)



TestWheels(True)
TestWheels(True)
TestWheels(True)
TestWheels(True)

TestWheels(True)
TestWheels(True)
TestWheels(True)
TestWheels(True)

TestWheels(True)
TestWheels(True)
TestWheels(True)
TestWheels(True)

TestWheels(True)
TestWheels(True)
TestWheels(True)
TestWheels(True)















