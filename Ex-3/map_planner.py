from typing import NamedTuple

import numpy as np

from map.occupancy_grid_map import OccupancyGridMap

"""
Imagine if `samples` were a array of List[NamedTuple]

class RobotState(NamedTuple):
    x: np.float32
    y: np.float32
    angle: np.float32
"""


def motion_model(samples: np.ndarray):
    pass


def observation_model(samples: np.ndarray):
    pass


def replace_with_uniform(samples: np.ndarray, N: int, map: OccupancyGridMap):
    """replance the last N samples with uniformly distributed ones"""
    pass


def initial_samples(N: int, map: OccupancyGridMap):
    samples = np.zeros_like(N, dtype=np.float32)
    replace_with_uniform(samples, N, map)
    return samples


def resample(samples_old: np.ndarray, weights_old: np.ndarray):
    k = len(weights_old)
    samples, weights_old = samples_old[weights_old > 0], weights_old[weights_old > 0]

    # handle edge case, when all weights are 0:
    if len(weights_old) == 0:
        return samples

    # normalize weights (so CDF works out):
    weights_old /= np.sum(weights_old)
    weights_cdf = np.cumsum(weights_old)

    # generate k samples of [0;1)
    rs = np.random.uniform(size=k)

    # find the (maximum) indicies, such that weights_cdf[indicies] <= rs:
    indices = np.searchsorted(weights_cdf, rs, side="left")

    return samples[indices]


# def simple_planner():
#     while True:
#
