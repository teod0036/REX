import matplotlib.pyplot as plt
import numpy as np

from map.occupancy_grid_map import OccupancyGridMap, draw_map


def load_array(name: str) -> np.ndarray:
    datafile_name = f"{name}.npy"

    with open(datafile_name, "rb") as f:
        return np.load(f)


map = OccupancyGridMap(low=np.array((-1, 0)), high=np.array((1, 2)), resolution=0.05)

plt.clf()
draw_map(
    load_array("map_test_data"),
    map.extent,
)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()
