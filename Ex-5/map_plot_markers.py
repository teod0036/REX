import os
import sys
from time import perf_counter, sleep
from typing import Dict, List, Tuple

import cv2
import numpy as np

from map.occupancy_grid_map import OccupancyGridMap

camera_offset_m = np.array((0, 22.5 / 100))

marker_half_depth_m = np.array(11 / 100)
marker_radius_m = np.array(18 / 100) + camera_offset_m[1]

try:
    from robot_extended import Marker

    def eprint(*args, **kwargs):
        print(f"{__name__}.py: ", *args, file=sys.stderr, **kwargs)


    def plot_markers(
        map: OccupancyGridMap, markers: List[Marker], plot=True
    ) -> Tuple[OccupancyGridMap, Dict[int, np.ndarray]]:
        if len(markers) == 0:
            return map, {}

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

        if plot:
            map.plot_centroid(centroid_pos, centroid_radius_sq)

        return map, {id: pos for (id, _), pos in zip(markers, centroid_pos)}
except:
    pass
