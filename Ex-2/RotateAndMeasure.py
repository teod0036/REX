from time import perf_counter, sleep

import robot
from rightSpeedModifier import rightSpeedmodifier

arlo = robot.Robot()

print(arlo.go_diff(0, 44, 0, 1))
sleep(2)
print(arlo.stop())
sleep(4)
print(arlo.go_diff(0, 44, 0, 1))
sleep(2)
print(arlo.go_diff(0, 52, 0, 1))
sleep(2)
print(arlo.go_diff(0, 60, 0, 1))
sleep(2)
print(arlo.go_diff(0, 72, 0, 1))
sleep(2)
print(arlo.go_diff(0, 80, 0, 1))
sleep(2)
print(arlo.go_diff(0, 88, 0, 1))
sleep(2)
print(arlo.go_diff(0, 96, 0, 1))
sleep(2)
print(arlo.go_diff(0, 104, 0, 1))
sleep(2)
print(arlo.go_diff(0, 112, 0, 1))
sleep(2)
print(arlo.go_diff(0, 120, 0, 1))
sleep(2)
print(arlo.stop())
sleep(1)
