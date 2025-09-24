from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from robot_extended import Marker, Pose, RobotExtended, eprint

base_resolution = 0.01  # meters per cell at LOD 0
robot_length_m = 1.45


def create_map(markers: List[Marker]):
    xz_tvec = -1 * np.array(
        [(pose.tvec[0], pose.tvec[2]) for _, pose in markers], dtype=np.float32
    )
    max_extent_m = np.max(np.abs(xz_tvec))  # scalar: max |x| or |z|

    LOD = np.ceil(0.5 * np.log2(2 * max_extent_m / (base_resolution * base_resolution)))
    resolution = base_resolution * (2**LOD)  # meter / cell

    map_size = max(1, int(np.ceil((2 * max_extent_m) / resolution)))
    map_array = np.full((map_size, map_size), 0, dtype=np.int32)
    center_i = map_size // 2
    center_j = map_size // 2

    map_array[center_i, center_j] = -1

    for id, (x, z) in zip([id for id, _ in markers], xz_tvec):
        i = int(center_i + (z / resolution))
        j = int(center_j - (x / resolution))

        if 0 <= i < map_size and 0 <= j < map_size:
            map_array[i, j] = id
        else:
            eprint(f"couldn't plot: {xz_tvec = }, {i = }, {j = }, {map_size = }")

    return map_array


# def save_map(map_array: np.ndarray, center_i: int, center_j: int):
#     import matplotlib.pyplot as plt
#
#     plt.figure(figsize=(8, 8))
#     plt.imshow(map_array, cmap="gray", origin="upper")
#     plt.title("Camera-centered map")
#     plt.xlabel("Z (Forward)")
#     plt.ylabel("X (Right)")
#
#     # Plot camera center
#     plt.scatter([center_j], [center_i], color="red", label="Camera Center", s=50)
#
#     # Set limits to center the camera in view
#     view_radius = 50  # how many pixels around the center to show (tweak as needed)
#     plt.xlim(center_j - view_radius, center_j + view_radius)
#     plt.ylim(center_i - view_radius, center_i + view_radius)
#
#     # Flip Y axis if necessary (image vs. Cartesian coords)
#     plt.gca().invert_yaxis()
#
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("robot_map.png")
#     print("Saved: robot_map.png")
#     plt.close()


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
    # map = create_map(markers)
    # save_map(map, map.size // 2, map.size // 2)
    print(create_map(RobotExtended().perform_image_analysis()))
