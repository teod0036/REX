# Arlo Robot Controller

import sys
import time
from typing import List, Optional, Tuple, NamedTuple

import cv2  # Import the OpenCV library
import numpy as np
import numpy.typing as npt

from robot import Robot

class PoseMarkers(NamedTuple):
    rvecs : npt.NDArray[np.float64]
    tvecs :  npt.NDArray[np.float64]
    objPoints :  npt.NDArray[np.float64]



def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def detectMarkers(
    image: npt.NDArray[np.uint8],
    dictionary: cv2.aruco.Dictionary,
    parameters: Optional[cv2.aruco.DetectorParameters] = None,
) -> Tuple[
    List[npt.NDArray[np.float32]],  # corners
    Optional[npt.NDArray[np.int32]],  # ids
    Optional[List[npt.NDArray[np.float32]]],  # rejectedImgPoints
]:
    return cv2.aruco.detectMarkers(image, dictionary, parameters)


def estimatePoseSingleMarkers(
    corners: npt.NDArray[np.float32],  # shape: (N, 1, 4, 2)
    markerLength: float,
    cameraMatrix: npt.NDArray[np.float64],  # shape: (3, 3)
    distCoeffs: npt.NDArray[np.float64],  # shape: (4,) or similar
) -> Tuple[
    npt.NDArray[np.float64],  # rvecs, shape: (N, 1, 3)
    npt.NDArray[np.float64],  # tvecs, shape: (N, 1, 3)
    npt.NDArray[np.float64],  # objPoints, shape: (N, 4, 3)
]:
    return cv2.aruco.estimatePoseSingleMarkers(
        corners, markerLength, cameraMatrix, distCoeffs
    )

class RobotExtended:
    FOCAL_LENGTH = 1257
    IMAGE_SIZE = (1640, 1232)
    DISTORTION_COEFFICENTS = np.zeros((5, 1))
    CAMERA_MATRIX = np.array(
        [
            [FOCAL_LENGTH, 0, IMAGE_SIZE[0] / 2],
            [0, FOCAL_LENGTH, IMAGE_SIZE[1] / 2],
            [0, 0, 1],
        ]
    )
    MARKER_LENGTH_M = 0.145

    def __init__(self, port="/dev/ttyACM0"):
        self.robot = Robot(port)
        try:
            import picamera2

            eprint("Camera.py: Using picamera2 module")
        except ImportError:
            eprint("Camera.py: picamera2 module not available")
            exit(-1)

        self.camera = picamera2.Picamera2()
        self.camera.configure(
            self.camera.create_still_configuration(
                {"size": self.IMAGE_SIZE, "format": "RGB888"},
                controls={"ScalerCrop": (0, 0, 3280, 2464)},
                queue=False,
            )
        )
        self.camera.start(show_preview=False)
        time.sleep(1)  # wait a little for camera setup

    def take_picture(self) -> npt.NDArray[np.uint8]:
        eprint("Taking picture...")
        return self.camera.capture_array("main")

    def perform_image_analysis(self):
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        corners_list, ids, rejectedimgpoints = detectMarkers(self.take_picture(), d)
        corners: np.ndarray = np.array(corners_list, dtype=np.float32)

        if ids is None:
            eprint("No landmarks found")
            return None
        else:
            eprint("Found landmarks: ", *ids)

        rvecs, tvecs, objPoints = estimatePoseSingleMarkers(
            corners,
            self.MARKER_LENGTH_M,
            self.CAMERA_MATRIX,
            self.DISTORTION_COEFFICENTS,
        )
        return PoseMarkers(rvecs, tvecs, objPoints)


eprint(RobotExtended().perform_image_analysis())
