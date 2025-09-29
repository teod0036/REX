
from map.occupancy_grid_map import OccupancyGridMap,draw_map
import matplotlib.pyplot as plt
import numpy as np

def load_array(name: str) -> np.ndarray:
    datafile_name = f"{name}.npy"

    with open(datafile_name, "rb") as f:
        return np.load(f)

plt.clf()
draw_map(load_array("map_test_data"), OccupancyGridMap().extent)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
