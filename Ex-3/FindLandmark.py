import datetime
from time import sleep

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

    for cnt in corners:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image, f"{w} x {h}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
    dt = datetime.datetime.now()
    cv2.imwrite(f"{dt.strftime('%M%S')}.jpeg", image)

    if ids is not None:
        for i in ids:
            print("FindLandmark.py: Found a landmark")

    for cnt in corners:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image, f"{w} x {h}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
    dt = datetime.datetime.now()
    cv2.imwrite(f"{dt.strftime('%M%S')}.jpeg", image)


perform_Findlandmark()
