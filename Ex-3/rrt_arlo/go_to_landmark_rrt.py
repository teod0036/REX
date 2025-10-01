import sys
from typing import List

import cv2
import numpy as np

from map.occupancy_grid_map import OccupancyGridMap
from robot_extended import Marker, Pose, RobotExtended, save_array

from plan_path import plan_path 
from exec_arlo_instructions import exec_instructions
import map.robot_models as robot_models
import map_plot_markers as map_plot_markers

map_low = map_plot_markers.map_low
map_high = map_plot_markers.map_low
map_res = map_plot_markers.map_res

marker_radius = map_plot_markers.marker_radius_m

def eprint(*args, **kwargs):
    print(f"{__name__}.py: ", *args, file=sys.stderr, **kwargs)




if __name__ == "__main__":
    # markers = [
    #     Marker(
    #         id=6,
    #         pose=Pose(
    #             rvec=np.array([-0.02632951, -3.09577251, 0.05794748]),
    #             tvec=np.array([0.12359971, 0.0692874, 0.40603545]),
    #             objPoint=np.array([-0.0725, 0.0725, 0.0], dtype=np.float32),
    #             corners=np.array(
    #                 [
    #                     [
    #                         [1439.0, 1058.0],
    #                         [983.0, 1058.0],
    #                         [972.0, 610.0],
    #                         [1424.0, 602.0],
    #                     ]
    #                 ],
    #                 dtype=np.float32,
    #             ),
    #         ),
    #     )
    # ]

    markers = RobotExtended().perform_image_analysis()
    # print(markers)
    map = OccupancyGridMap(low=map_low, high=map_high, resolution=map_res)
    map.clear()
    map, landmarks = map_plot_markers.plot_markers(map, markers, plot=False)
    save_array(map.grid, "map_test_data")

    robot = robot_models.PointMassModel(ctrl_range=[-map.resolution, map.resolution])
    
    target_landmark = 6
    #goal = (landmarks[target_landmark][0], landmarks[target_landmark][1]-marker_radius-(map_res*2))
    #print(goal)
    instructions = plan_path(map=map, robot=robot, debug=True)
    
    if len(instructions) != 0:  
        exec_instructions(instructions)
    # import matplotlib.pyplot as plt
    # plt.clf()
    # map.draw_map()
    # plt.show()
