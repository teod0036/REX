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

def gridize_map(landmark_coordinates, map_size, cell_size):
    grid_map = np.zeros((map_size, map_size), dtype=int)
    grid_map[int((map_size-1)/2)][int((map_size-1)/2)] = 99

    for mark in landmark_coordinates:
        x = int(np.round(mark[1]*100/cell_size)) + int((map_size-1)/2)
        y = int(np.round(mark[2]*100/cell_size)) + int((map_size-1)/2)
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

        grid_map[y][x] = int(mark_id)

    return grid_map

    #eprint(grid_map)

def collision(grid_map, map_size, cell_size, grid_position, obstacle_coordinates):
    x, y = grid_position
    #check on the position
    if grid_map[x][y] != 0 or grid_map[x][y] != 99:
        return True
    
    return detailed_collision(map_size, cell_size, (x, y), obstacle_coordinates)
            

def detailed_collision(map_size, cell_size, grid_position, obstacle_coordinates):
    arlo_radius = 22.5
    x, y = grid_position
    x_grid = (y-map_size)/100*cell_size
    y_grid = (x-map_size)/100*cell_size

    for mark in obstacle_coordinates:
        x_mark = mark[1]
        y_mark = mark[2]
        if np.sqrt(np.square(x_grid-x_mark) + np.square(y_grid-y_mark)) <= arlo_radius:
            return True

    return False

gridize_map(get_map(), 25, 40)
