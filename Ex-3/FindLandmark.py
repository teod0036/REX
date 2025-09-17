from time import sleep
import robot
from Turn90 import perform_Turn90
from imagecapture import takePicture
import cv2

arlo = robot.Robot()

def perform_Findlandmark():
    cv2.aruco.DICT_6x6_250
    cv2.aruco.ArucoDetector.detectMarkers