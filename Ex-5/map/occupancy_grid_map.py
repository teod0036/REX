from typing import Tuple

import numpy as np

if __name__ == "__main__":
    from aabb import AABB
else:
    from map.aabb import AABB


class OccupancyGridMap:
    def __init__(
        self,
        low=np.array((0, 0), dtype=np.float32),
        high=np.array((2, 2), dtype=np.float32),
        resolution=0.05,
    ) -> None:
        """
        note: low included, high not include in grid
        """

        self.aabb = AABB(low, high)
        self.resolution = resolution
        self.grid_size = (self.aabb.size / resolution).astype(int)
        self.grid_x, self.grid_y = self.grid_size
        self.grid = np.zeros(self.grid_size, dtype=np.uint8)
        self.meshgrid = np.meshgrid(
            np.arange(self.grid_x), np.arange(self.grid_y), indexing="ij"
        )  # shape (grid_x, grid_y)
        self.centroids = self.aabb.min + self.resolution * (
            np.stack((self.meshgrid[0] + 0.5, self.meshgrid[1] + 0.5), axis=-1)
        )  # shape (grid_x, grid_y, 2)

    def clear(self):
        self.grid = np.zeros(self.grid_size, dtype=np.uint8)

    @property
    def map_area(self):
        return np.array((self.aabb.min, self.aabb.max))

    @property
    def map_size(self):
        return self.aabb.size

    @property
    def extent(self):
        return (
            self.aabb.left,
            self.aabb.right,
            self.aabb.bottom,
            self.aabb.top,
        )

    def draw_map(self, ax=None):
        import matplotlib.pyplot as plt

        plt.imshow(
            self.grid.transpose(),
            cmap="Greys",
            origin="lower",
            vmin=0,
            vmax=1,
            extent=self.extent,
            interpolation="none",
        )
        plt.show()

    def populate(self, n_obs=6):
        """
        generate a grid map with some circle shaped obstacles
        """
        origins = np.random.uniform(
            low=self.aabb.min + self.map_size[0] * 0.2,
            high=self.aabb.min + self.map_size[0] * 0.8,
            size=(n_obs, 2),
        )
        radius = np.random.uniform(low=0.1, high=0.3, size=n_obs)
        radius_sq = radius**2
        radius_sq_reshaped = radius_sq[np.newaxis, np.newaxis, :]  # shape (_, _, N)

        return self.plot_centroid(origins, radius_sq_reshaped)

    def plot_centroid(self, origins: np.ndarray, radius_squared: np.ndarray):
        # fill the grids by checking if the grid centroid is in any of the circle

        centorids_reshaped = self.centroids[
            :, :, np.newaxis, :
        ]  # shape (grid_x, grid_y, _, 2)
        origins_reshaped = origins[
            np.newaxis, np.newaxis, :, :
        ]  # shape (     _,      _, N, 2)

        dist = centorids_reshaped - origins_reshaped  # shape (grid_x, grid_y, N, 2)
        dist_squared = np.sum(dist**2, axis=-1)  # shape (grid_x, grid_y, N)

        mask = np.any(dist_squared <= radius_squared, axis=-1).reshape(
            (self.grid_x, self.grid_y)
        )  # shape (grid_x, grid_y)

        self.grid[mask] = 1

    def in_collision(self, pos):
        """
        find if the position is occupied or not. return if the queried pos is outside the map
        """
        indices = np.rint((pos - self.aabb.min) / self.resolution).astype(int)
        if np.any((indices < np.array([0, 0])) | (self.grid_size <= indices)):
            return 1
        else:
            return self.grid[indices[0], indices[1]]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    map = OccupancyGridMap()
    map.populate()
    plt.clf()
    map.draw_map()
    plt.show()
