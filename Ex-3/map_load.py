import matplotlib.pyplot as plt
import numpy as np

from map.occupancy_grid_map import OccupancyGridMap, draw_map


def load_array(name: str) -> np.ndarray:
    datafile_name = f"{name}.npy"

    with open(datafile_name, "rb") as f:
        return np.load(f)


gridmap = OccupancyGridMap(
    low=np.array((0, 0)), high=np.array((2000, 2000)), resolution=50
)

plt.clf()
draw_map(
    load_array("map_test_data"),
    gridmap.extent,
)
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.xlim(
    left=-gridmap.aabb.center[0], right=gridmap.extent[1][0] - gridmap.aabb.center[0]
)
plt.show()
