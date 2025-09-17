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

    print("FindLandmark.py: Fetching the dictionary")

    dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    print("FindLandmark.py: Attempting to detect Markers")

    corners,ids,rejectedimgpoints = cv2.aruco.detectMarkers(image,dict)

    if (ids is not None):
        for i in ids:
            print("FindLandmark.py: Found a landmark")

perform_Findlandmark()