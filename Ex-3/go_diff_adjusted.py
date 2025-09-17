from robot import Robot

def go_diff_adjusted(robot: Robot, powerLeft: int, powerRight: int, dirLeft: int, dirRight: int):
   powerLeftNew = max(43, powerLeft)
   powerRightNew = max(43, powerRight) - 2

   if dirLeft == 1 and dirRight == 1:
	powerLeftNew = powerLeftNew if powerLeft != 0 else 0
	powerRightNew = round(powerRightNew - 0.4 * (powerRightNew - 41)) if powerRight != 0 else 0

	robot.go_diff(powerLeftNew, powerRightNew, dirLeft, dirRight)