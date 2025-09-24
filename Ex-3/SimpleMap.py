import robot_extended
import numpy as np
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

arlo_master = robot_extended.RobotExtended()
arlo = arlo_master.robot

def get_map():
    landmarks = arlo_master.perform_image_analysis()
    
    landmark_coordinates = [[]]
    i = 0
    for marker in landmarks:
        landmark_coordinates[i] = [marker.id, marker.pose.tvec[0], marker.pose.tvec[2]]
        i += 1
    
    return landmark_coordinates

def visualize_map(landmark_coordinates):
    map_size = 13
    visual_map = np.zeros((map_size, map_size), dtype=int)
    visual_map[int((map_size-1)/2)][int((map_size-1)/2)] = 1
    eprint(visual_map)

visualize_map(get_map())
