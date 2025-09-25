from typing import List

import numpy as np
from robot_extended import Marker, Pose, RobotExtended, load_array, save_array

from occupancy_grid_map import OccupancyGridMap,draw_map

marker_half_depth = 0.70  # meter
marker_radius = 1.4


def create_local_map(markers: List[Marker]):
    tvecs = np.array(
        [pose.tvec for _, pose in markers], dtype=np.float32
    )  # shape (N, 3)
    rvecs = np.array(
        [pose.rvec for _, pose in markers], dtype=np.float32
    )  # shape (N, 3)
    xz_tvec = -tvecs[:, [0, 2]]  # shape (N, 2)
    xz_rvec = +rvecs[:, [0, 2]]  # shape (N, 2)

    def normalize(v):
        norm = np.linalg.norm(v)
        return v / norm

    if len(xz_tvec) > 0:
        marker_center = xz_tvec - normalize(xz_rvec[:, 1]) * marker_half_depth

        map = OccupancyGridMap()
        map.plot_centroid(marker_center, marker_radius * np.ones_like(marker_center))
        save_array(map.grid, "map_grid")


if __name__ == "__main__":
    # markers = [
    #     Marker(
    #         id=1,
    #         pose=Pose(
    #             rvec=np.array([3.07823555, 0.00605985, 0.42181049], dtype=np.float32),
    #             tvec=np.array([-0.09892818, 0.08039518, 0.79364262], dtype=np.float32),
    #             objPoint=np.array([-0.0725, 0.0725, 0.0], dtype=np.float32),
    #             corners=np.array(
    #                 [[[547.0, 628.0], [775.0, 629.0], [775.0, 854.0], [545.0, 864.0]]],
    #                 dtype=np.float32,
    #             ),
    #         ),
    #     )
    # ]

    markers = RobotExtended().perform_image_analysis()
    print(markers)

    create_local_map(markers)

