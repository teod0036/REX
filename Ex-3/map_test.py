from typing import List, NamedTuple

import numpy as np

from map.occupancy_grid_map import OccupancyGridMap, draw_map
from map.transform import AABB
from robot_extended import RobotExtended, save_array


class Pose(NamedTuple):
    rvec: np.ndarray
    tvec: np.ndarray
    objPoint: np.ndarray
    corners: np.ndarray


class Marker(NamedTuple):
    id: int
    pose: Pose


marker_half_depth_m = 0.70  # meter
marker_radius_m = 1.4
cell_size_cm = 40
# low_ma
# AABB = AABB(


def create_local_map(markers: List[Marker]):
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

    if len(xz_tvec) > 0:
        map = OccupancyGridMap()

        marker_center = xz_tvec - normalize(xz_rvec[:, 1]) * marker_half_depth_m
        scale = 100 / cell_size_cm

        map.plot_centroid(marker_center * scale + map.resolution * map.grid_size // 2,
            np.ones_like(marker_center) * marker_radius_m * scale,
        )
        # # print(marker_center * 100 * map.resolution + map.grid_size // 2)
        #
        # import matplotlib.pyplot as plt

        # plt.clf()
        # map.draw_map()
        # plt.show()
        save_array(map.grid, "map_test_data")


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
    create_local_map(markers)
