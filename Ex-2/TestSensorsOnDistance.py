from time import perf_counter, time
import robot
from Turn90 import perform_Turn90

arlo = robot.Robot()

newlist = []
def test_sensors():

    for i in range(5):
        resultval = arlo.read_front_ping_sensor()
        newlist.append(resultval)
        time.sleep(1)
    print(newlist)

test_sensors()