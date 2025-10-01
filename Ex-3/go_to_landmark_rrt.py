import sys
from typing import List

import cv2
import numpy as np

from rrt_arlo.map.occupancy_grid_map import OccupancyGridMap
from robot_extended import Marker, Pose, RobotExtended, save_array

from rrt_arlo.plan_path import plan_path 
from rrt_arlo.exec_arlo_instructions import exec_instructions
import rrt_arlo.map.robot_models as robot_models
import rrt_arlo.map_plot_markers as map_plot_markers

from typing import Tuple, Dict
map_low = np.array((-1, 0))
map_high = np.array((1, 2))
map_res = 0.05


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
    map, landmarks = map_plot_markers.plot_markers(map, markers)
    save_array(map.grid, "map_test_data")

    robot = robot_models.PointMassModel(ctrl_range=[-map.resolution, map.resolution])

    instructions = plan_path(map=map, robot=robot, goal=landmarks[6], debug=True)

    exec_instructions(instructions)
    # import matplotlib.pyplot as plt
    # plt.clf()
    # map.draw_map()
    # plt.show()
