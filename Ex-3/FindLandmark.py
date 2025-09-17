import datetime
from time import sleep

import cv2
import numpy as np

import robot
from imagecapture import takePicture
from Turn90 import perform_Turn90

arlo = robot.Robot()


def CreateCameraMatrix(image):
    focallength = 1257
    imageheight = image.shape[1]
    imagewidth = image.shape[0]
    print(
        "FindLandmark.py: Image Height: "
        + str(imageheight)
        + " Image Width: "
        + str(imagewidth)
    )

    cameramatrix = np.array(
        [[focallength, 0, imagewidth / 2], [0, focallength, imageheight / 2], [0, 0, 1]]
    )
    return cameramatrix


def perform_Findlandmark():
    print("FindLandmark.py: Taking a picture using imagecapture")

    image = takePicture()
    
    print("FindLandmark.py: Fetching the dictionary")

    dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    print("FindLandmark.py: Attempting to detect Markers")

    corners, ids, rejectedimgpoints = cv2.aruco.detectMarkers(image, dict)
    for cnt in corners:
        print("printing corner" + str(cnt))
    CreateDetectionImage(corners, rejectedimgpoints, image)

    if ids is None:
        print("FindLandmark.py: No landmarks found, ending FindLandmark")
        dt = datetime.datetime.now()
        cv2.imwrite(f"Test123{dt.strftime('%M%S')}.jpeg", image)
        print(f"outputted to Test123{dt.strftime('%M%S')}.jpeg")
        return None
    else:
        for i in ids:
            print("FindLandmark.py: Found landmark ID" + str(id))
    
    cameramatrix = CreateCameraMatrix(image)

    print("FindLandmark.py: Attempting to estimatePoseSingleMarkers")
    distcoefficients = np.zeros((5, 1))
    rotationvectors, translationvectors, objpoints = (
        cv2.aruco.estimatePoseSingleMarkers(
            corners, 0.145, cameramatrix, distcoefficients
        )
    )
    alllandmarksdict = {}
    y=0
    while y < len(ids):
        alllandmarksdict.update({int(ids[y]):CreateLandMarkArray(ids[y],rotationvectors[y],translationvectors[y],objpoints[y])})
        y+=1
    

    for tvec, rvec in zip(translationvectors, rotationvectors):
        cv2.drawFrameAxes(image, cameramatrix, distcoefficients, rvec, tvec, 0.1)

    dt = datetime.datetime.now()
    cv2.imwrite(f"Test123{dt.strftime('%M%S')}.jpeg", image)
    print(f"outputted to Test123{dt.strftime('%M%S')}.jpeg")

    for x in alllandmarksdict:
        print(str(x))
    return alllandmarksdict


def CreateLandMarkArray(id, rotationvectors, translationvectors, objpoints):
    print("----------------------------------------------------------------")
    print("FindLandmark.py: Creating landmarkarray with the following data")
    print("Rotationvectors: " + str(rotationvectors))
    print("Translationvectors (Horizontal,Vertical,Distance): " + str(translationvectors))
    print("Objpoints:" + str(objpoints))
    print("-------------------------------------------------------------")

    landmark = [rotationvectors,translationvectors,objpoints]
    return landmark


def CreateDetectionImage(corners, rejectedimgpoints, image):
    print("FindLandmark.py: Attempting to save picture with detection boxes on")
    for cnt in corners:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image, f"{w} x {h}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
    for cnt in rejectedimgpoints:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)


perform_Findlandmark()
