# Arlo Robot Controller

import datetime
import sys
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

import cv2  # Import the OpenCV library
import numpy as np

from robot import Robot


class Pose(NamedTuple):
    rvec: np.ndarray
    tvec: np.ndarray
    objPoint: np.ndarray
    corners: np.ndarray


class Marker(NamedTuple):
    id: int
    pose: Pose


def eprint(*args, **kwargs):
    print("robot_extended.py: ", *args, file=sys.stderr, **kwargs)


def detectMarkers(
    image: np.ndarray,
    dictionary: cv2.aruco.Dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_6X6_250
    ),
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


def save_picture(image: np.ndarray, prefix: str = "Test", suffix: Optional[str] = None):
    dt = datetime.datetime.now()
    image_name = f"{prefix}{dt.strftime('%M%S') if not suffix else suffix}.jpeg"

    cv2.imwrite(image_name, image)
    eprint(f"outputted to {image_name}")


class RobotExtended:
    FOCAL_LENGTH = 1257
    IMAGE_SIZE = (1640, 1232)
    DISTORTION_COEFFICENTS = np.array([0, 0, 0, 0, 0])
    CAMERA_MATRIX = np.array(
        [
            [FOCAL_LENGTH, 0, IMAGE_SIZE[0] / 2],
            [0, FOCAL_LENGTH, IMAGE_SIZE[1] / 2],
            [0, 0, 1],
        ]
    )
    MARKER_LENGTH_METER = 0.145

    def __init__(self, port="/dev/ttyACM0"):
        self.robot = Robot(port)
        try:
            import picamera2

            eprint("Using picamera2 module")
        except ImportError:
            eprint("picamera2 module not available")
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

    def take_detection_picture(self) -> np.ndarray:
        image: np.ndarray = self.take_picture()
        markers: List[Marker] = self.perform_image_analysis(image)
        for id, pose in markers:
            x, y, w, h = cv2.boundingRect(pose.corners)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{id}: {w} x {h}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            for tvec, rvec in zip(pose.tvec, pose.rvec):
                cv2.drawFrameAxes(
                    image,
                    self.CAMERA_MATRIX,
                    self.DISTORTION_COEFFICENTS,
                    rvec,
                    tvec,
                    0.1,
                )
        return image

    def perform_image_analysis(self, image : Optional[np.ndarray] = None) -> List[Marker]:
        corners_list, ids, _ = detectMarkers(
            self.take_picture() if not image else image
        )
        corners = np.array(corners_list, dtype=np.float32)

        if ids is None or len(ids) == 0:
            eprint("No landmarks found")
            return []
        else:
            eprint("Found landmarks: ", *ids)

        rvecs, tvecs, objPoints = estimatePoseSingleMarkers(
            corners,
            self.MARKER_LENGTH_METER,
            self.CAMERA_MATRIX,
            self.DISTORTION_COEFFICENTS,
        )
        return [
            Marker(
                int(ids[i]),
                Pose(
                    rvecs[i].reshape(3),
                    tvecs[i].reshape(3),
                    objPoints[i].reshape(3),
                    corners[i],
                ),
            )
            for i in range(len(ids))
        ]

    def perform_image_analysis_table(self) -> Dict[int, Pose]:
        return {i: pose for i, pose in self.perform_image_analysis()}


if __name__ == "__main__":
    save_picture(RobotExtended().take_detection_picture())
