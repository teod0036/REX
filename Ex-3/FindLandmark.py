import datetime
from time import sleep
import numpy as np
import cv2

import robot
from imagecapture import takePicture
from Turn90 import perform_Turn90

arlo = robot.Robot()


def perform_Findlandmark():
    print("FindLandmark.py: Taking a picture")

    image = takePicture()
    print("FindLandmark.py: Saved the picture, ")

    print("FindLandmark.py: Fetching the dictionary")

    dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    print("FindLandmark.py: Attempting to detect Markers")

    corners, ids, rejectedimgpoints = cv2.aruco.detectMarkers(image, dict)

    if ids is not None:
        for i in ids:
            print("FindLandmark.py: Found a landmark")

    for cnt in corners:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image, f"{w} x {h}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
    for cnt in rejectedimgpoints:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    print("FindLandmark.py: Attempting to save picture with detection boxes on")

    dt = datetime.datetime.now()
    cv2.imwrite(f"{dt.strftime('%M%S')}.jpeg", image)

    focallength = 1284
    imageheight = 1080
    imagewidth = 1920
    distcoefficients = np.zeros((5,1))
    cameramatrix = np.array([[focallength,    0,              imagewidth/2],
                    [0,              focallength,    imageheight/2],
                    [0,              0,              1]])

    print("FindLandmark.py: Attempting to estimatePoseSingleMarkers")
    rotationvectors,translationvectors,objpoints = cv2.aruco.estimatePoseSingleMarkers(corners,0.145,cameramatrix,distcoefficients)

    if ids is not None:
        print("FindLandmark.py: printing rotation vectors")
        for y in rotationvectors:
            print(y)
        print("FindLandmark.py: printing translation vectors")
        for z in translationvectors:
            print(z)



perform_Findlandmark()
