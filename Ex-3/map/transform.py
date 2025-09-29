import numpy as np


class Transform2D:
    def __init__(self, mat=np.eye(3)) -> None:
        self.mat = mat

    def stack(self, mat: np.matrix):
        self.mat = np.matmul(mat, self.mat)
        return self

    def apply(self, vec: np.ndarray):
        ret = np.matmul(self.mat, np.array((vec[2], 1)))

        return np.array(ret[0:2])

    def apply_to_dir(self, vec: np.ndarray):
        ret = np.matmul(self.mat, np.array((vec[2], 0)))

        return np.array(ret[0:2])

    def translate(self, delta: np.ndarray):
        mat = np.matrix([[1, 0, delta[0]], [0, 1, delta[1]], [0, 0, 1]])

        return self.stack(mat)

    def rotate(self, angle: float):
        sin_angle = np.sin(angle)
        cos_angle = np.cos(angle)

        mat = np.matrix(
            [[cos_angle, sin_angle, 0], [-sin_angle, cos_angle, 0], [0, 0, 1]],
            dtype=np.float32,
        )

        return self.stack(mat)

    def scale(self, scale):
        mat = np.matrix([[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]])

        return self.stack(mat)

    def reflectX(self):
        return self.scale(np.array([-1, +1]))

    def reflecty(self):
        return self.scale(np.array([+1, -1]))

    def sheerX(self, s: float):
        mat = np.matrix([[1, 0, 0], [s, 1, 0], [0, 0, 1]])

        return self.stack(mat)

    def sheerY(self, t: float):
        mat = np.matrix([[1, t, 0], [0, 1, 0], [0, 0, 1]])

        return self.stack(mat)

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

    def to_transform(self):
        """
        matrix that transforms points from the unit square to the AABB
        """
        return Transform2D().scale(self.size).translate(self.min)

    def to_transform_inv(self):
        """
        matrix that transforms points from AABB to the unit square
        """
        return Transform2D().translate(-self.min).scale(1 / self.size)
