from typing import NamedTuple

import numpy as np


class RobotState(NamedTuple):
    x: np.float32
    y: np.float32
    angle: np.float32


