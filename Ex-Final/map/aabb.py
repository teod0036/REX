import numpy as np


class AABB:
    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        self.min = low
        self.max = high

    @property
    def center(self):
        return (self.max + self.min) / 2

    @property
    def size(self):
        return self.max - self.min

    @property
    def left(self):
        return self.min[0]

    @property
    def bottom(self):
        return self.min[1]

    @property
    def right(self):
        return self.max[0]

    @property
    def top(self):
        return self.max[1]
