from time import sleep
import robot
from Turn90 import perform_Turn90
from imagecapture import takePicture
import cv2

arlo = robot.Robot()

def perform_Findlandmark():
    print("FindLandmark.py: Taking a picture")

    image = takePicture()
    print("FindLandmark.py: Saved the picture, ")

    print("FindLandmark.py: Attempting to detect Markers")

    ##vals = cv2.aruco.ArucoDetector.detectMarkers(image,cv2.aruco.DICT_6X6_250)

    vals = cv2.aruco.detectMarkers(image,cv2.aruco.DICT_6X6_250)

    print("FindLandmark.py: Result of markerdetection:" + vals[0])

perform_Findlandmark()