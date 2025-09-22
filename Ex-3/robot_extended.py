# Arlo Robot Controller

import datetime
import time
from pprint import pprint

import cv2  # Import the OpenCV library
import numpy as np

from robot import Robot


class RobotExtended:
    FOCAL_LENGTH = 1257
    IMAGE_SIZE = (1640, 1232)

    def __init__(self, port="/dev/ttyACM0"):
        self.robot = Robot(port)

    def take_picture(self):
        try:
            import picamera2

            print("Camera.py: Using picamera2 module")
        except ImportError:
            print("Camera.py: picamera2 module not available")
            exit(-1)

        cam = picamera2.Picamera2()
        cam.configure(
            cam.create_still_configuration(
                {"size": self.IMAGE_SIZE, "format": "RGB888"},
                controls={"ScalerCrop": (0, 0, 3280, 2464)},
                queue=False,
            )
        )
        cam.start(show_preview=False)

        time.sleep(1) # wait a little for camera setup

        return cam.capture_array("main")

    def camera_matrix(self):
        cameramatrix = np.array(
            [
                [self.FOCAL_LENGTH, 0, self.IMAGE_SIZE[0] / 2],
                [0, self.FOCAL_LENGTH, self.IMAGE_SIZE[1] / 2],
                [0, 0, 1],
            ]
        )
        return cameramatrix

    # def perform_image_analysis():
    #     return

print(RobotExtended().take_picture())
