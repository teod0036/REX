import robot_extended
import numpy as np
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

arlo_master = robot_extended.RobotExtended()
arlo = arlo_master.robot

def get_map():
    landmarks = arlo_master.perform_image_analysis()
    
    landmark_coordinates = []
    for marker in landmarks:
        landmark_coordinates.append([marker.id, marker.pose.tvec[0], marker.pose.tvec[2]])
    
    return landmark_coordinates

def gridize_map(landmark_coordinates):
    map_size = 13
    grid_map = np.zeros((map_size, map_size), dtype=int)
    grid_map[0][int((map_size-1)/2)] = 99

    for mark in landmark_coordinates:
        x = int(np.round(mark[1]*100/20)) + int((map_size-1)/2)
        y = int(np.round(mark[2]*100/20)) + int((map_size-1)/2)
        mark_id = mark[0]
        if x > 12:
            x = 12
            mark_id *= -1
        elif x < 0:
            x = 0
            mark_id *= -1
        if y > 12:
            y = 12
            if mark_id > 0:
                mark_id *= -1 
        elif y < 0:
            y = 0
            if mark_id > 0:
                mark_id *= -1
        
        eprint(f"landmark {mark_id} is at ({mark[1]},{mark[2]}) in world")
        eprint(f"landmark {mark_id} is at ({x},{y}) in map")
        grid_map[y][x] = int(mark_id)


    eprint(grid_map)

gridize_map(get_map())
