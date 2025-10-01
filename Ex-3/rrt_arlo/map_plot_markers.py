import sys
from time import perf_counter, sleep
from typing import List

import os


import cv2
import numpy as np

from map.occupancy_grid_map import OccupancyGridMap
from robot_extended import Marker, Pose, RobotExtended, save_array

marker_half_depth_m = np.array(11 / 100)
marker_radius_m = np.array(18 / 100)
camera_offset_m = np.array((0, 22.5 / 100))

map_low = np.array((-1, 0))
map_high = np.array((1, 2))
map_res = 0.05


def eprint(*args, **kwargs):
    print(f"{__name__}.py: ", *args, file=sys.stderr, **kwargs)


def create_local_map(map: OccupancyGridMap, markers: List[Marker]) -> OccupancyGridMap:
    if len(markers) == 0:
        return map

    tvecs = np.array(
        [pose.tvec for _, pose in markers], dtype=np.float32
    )  # shape (N, 3)
    raxes = np.array(
        [cv2.Rodrigues(pose.rvec)[0] for _, pose in markers], dtype=np.float32
    )  # shape (N, 3)

    pos = (
        tvecs[:, [0, 2]] - raxes[:, :, 2][:, [0, 2]] * marker_half_depth_m
    )  # shape (N, 2)

    centroid_pos = camera_offset_m + pos
    centroid_radius_sq = marker_radius_m**2

    eprint(f"{centroid_pos = }")
    eprint(f"{centroid_radius_sq = }")

    map.plot_centroid(centroid_pos, centroid_radius_sq)

    return map


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

    arlo_master = RobotExtended()
    map = OccupancyGridMap(low=map_low, high=map_high, resolution=map_res)

    while True:  # or some other form of loop
        markers = arlo_master.perform_image_analysis()
        create_local_map(map, markers)

        if os.path.exists("./map_data.npy"):
            os.remove("./map_data.npy")
        save_array(map.grid, "map_data")

        sleep(1)

    # import matplotlib.pyplot as plt
    # plt.clf()
    # map.draw_map()
    # plt.show()
