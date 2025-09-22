# Arlo Robot Controller

import sys
import time
from typing import List, NamedTuple, Optional, Tuple

import cv2  # Import the OpenCV library
import numpy as np

from robot import Robot


class Pose(NamedTuple):
    rvecs: np.ndarray
    tvecs: np.ndarray
    objPoints: np.ndarray


class Marker(NamedTuple):
    id: int
    pose: Pose


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def detectMarkers(
    image: np.ndarray,
    dictionary: cv2.aruco.Dictionary,
    parameters: Optional[cv2.aruco.DetectorParameters] = None,
) -> Tuple[
    List[np.ndarray],  # corners
    Optional[np.ndarray],  # ids
    Optional[List[np.ndarray]],  # rejectedImgPoints
]:
    return cv2.aruco.detectMarkers(image, dictionary, parameters)


def estimatePoseSingleMarkers(
    corners: np.ndarray,  # shape: (N, 1, 4, 2)
    markerLength: float,
    cameraMatrix: np.ndarray,  # shape: (3, 3)
    distCoeffs: np.ndarray,  # shape: (4,) or similar
) -> Tuple[
    np.ndarray,  # rvecs, shape: (N, 1, 3)
    np.ndarray,  # tvecs, shape: (N, 1, 3)
    np.ndarray,  # objPoints, shape: (N, 4, 3)
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

    def __del__(self):
        self.camera.close()

    def take_picture(self) -> np.ndarray:
        eprint("Taking picture...")
        return self.camera.capture_array("main")

    def perform_image_analysis(self) -> List[Marker]:
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        corners_list, ids, rejectedimgpoints = detectMarkers(self.take_picture(), d)
        corners: np.ndarray = np.array(corners_list, dtype=np.float32)

        if not ids:
            eprint("No landmarks found")
            return []
        else:
            eprint("Found landmarks: ", *ids)

        rvecs, tvecs, objPoints = estimatePoseSingleMarkers(
            corners,
            self.MARKER_LENGTH_M,
            self.CAMERA_MATRIX,
            self.DISTORTION_COEFFICENTS,
        )
        return [
            Marker(int(ids[i]), Pose(rvecs[i], tvecs[i], objPoints[i]))
            for i in range(len(ids))
        ]


if __name__ == "__main__":
    eprint(RobotExtended().perform_image_analysis())
