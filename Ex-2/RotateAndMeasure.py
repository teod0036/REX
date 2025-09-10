from time import perf_counter, sleep

import robot
from rightSpeedModifier import rightSpeedmodifier

arlo = robot.Robot()

print(arlo.go_diff(0, 43, 0, 1))
sleep(2)
print(arlo.stop())
sleep(5)
print(arlo.go_diff(0, 43, 0, 1))
sleep(5)
print(arlo.stop())
sleep(1)
