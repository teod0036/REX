from typing import List, NamedTuple

import numpy as np

from map.occupancy_grid_map import OccupancyGridMap, draw_map
from map.transform import AABB
from robot_extended import Marker, RobotExtended, save_array

marker_half_depth_m = 11 / 100  # meter
marker_radius_cm = 18
cell_size_cm = 8


def create_local_map(markers: List[Marker]) -> OccupancyGridMap:
    map = OccupancyGridMap()
    scale = cell_size_cm * (min(map.grid_x, map.grid_y))

    if len(markers) == 0:
        return map

    tvecs = np.array(
        [pose.tvec for _, pose in markers], dtype=np.float32
    )  # shape (N, 3)
    rvecs = np.array(
        [pose.rvec for _, pose in markers], dtype=np.float32
    )  # shape (N, 3)
    xz_tvec = -tvecs[:, [2, 0]]  # shape (N, 2)
    xz_rvec = +rvecs[:, [2, 0]]  # shape (N, 2)

    def normalize(v):
        norm = np.linalg.norm(v)
        return v / norm

    marker_center = xz_tvec - normalize(xz_rvec[:, 1]) * marker_half_depth_m

    centroid_pos = marker_center * 100 / scale
    centroid_radius = np.ones_like(marker_center) * marker_radius_cm / scale

    print(centroid_pos)

    map.plot_centroid(centroid_pos + map.aabb.center, centroid_radius)

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

    markers = RobotExtended().perform_image_analysis()
    print(markers)
    save_array(create_local_map(markers).grid, "map_test_data")

    # print(marker_center * 100 * map.resolution + map.grid_size // 2)
    # import matplotlib.pyplot as plt
    # plt.clf()
    # map.draw_map()
    # plt.show()
