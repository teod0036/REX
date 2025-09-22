import datetime
from time import sleep
import sys

import cv2
import numpy as np

import robot
from imagecapture import takePicture, initCamera
from Turn90 import perform_Turn90

arlo = robot.Robot()

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def CreateCameraMatrix(image):
    focallength = 1257
    imageheight = image.shape[1]
    imagewidth = image.shape[0]
    eprint(
        "FindLandmark.py: Image Height: "
        + str(imageheight)
        + " Image Width: "
        + str(imagewidth)
    )

    cameramatrix = np.array(
        [[focallength, 0, imagewidth / 2], [0, focallength, imageheight / 2], [0, 0, 1]]
    )
    return cameramatrix


def perform_Findlandmark(cam):
    eprint("FindLandmark.py: Taking a picture using imagecapture")

    image = takePicture(cam)

    eprint("FindLandmark.py: Fetching the dictionary")

    dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    eprint("FindLandmark.py: Attempting to detect Markers")

    corners, ids, rejectedimgpoints = cv2.aruco.detectMarkers(image, dict)
    for cnt in corners:
        eprint("eprinting corner" + str(cnt))

    if ids is None:
        eprint("FindLandmark.py: No landmarks found, ending FindLandmark")
        dt = datetime.datetime.now()
        cv2.imwrite(f"Test123{dt.strftime('%M%S')}.jpeg", image)
        eprint(f"outputted to Test123{dt.strftime('%M%S')}.jpeg")
        return None
    else:
        for i in ids:
            eprint("FindLandmark.py: Found landmark ID" + str(id))

    cameramatrix = CreateCameraMatrix(image)

    eprint("FindLandmark.py: Attempting to estimatePoseSingleMarkers")
    distcoefficients = np.zeros((5, 1))
    rotationvectors, translationvectors, objpoints = (
        cv2.aruco.estimatePoseSingleMarkers(
            corners, 0.145, cameramatrix, distcoefficients
        )
    )
    alllandmarksdict = {}
    for y in range(len(ids)):
        alllandmarksdict.update(
            {
                int(ids[y]): CreateLandMarkArray(
                    ids[y], rotationvectors[y], translationvectors[y], objpoints[y]
                )
            }
        )

    CreateDetectionImage(
        corners,
        rejectedimgpoints,
        image,
        translationvectors,
        rotationvectors,
        cameramatrix,
        distcoefficients,
    )

    for k, v in alllandmarksdict.items():
        eprint(f"key: {k}, value: {v}")

    return alllandmarksdict


def CreateLandMarkArray(id, rotationvectors, translationvectors, objpoints):
    eprint("----------------------------------------------------------------")
    eprint("FindLandmark.py: Creating landmarkarray with the following data")
    eprint("Rotationvectors: " + str(rotationvectors))
    eprint(
        "Translationvectors (Horizontal,Vertical,Distance): " + str(translationvectors)
    )
    eprint("Objpoints:" + str(objpoints))
    eprint("-------------------------------------------------------------")

    landmark = [rotationvectors, translationvectors, objpoints]
    return landmark


def CreateDetectionImage(
    corners,
    rejectedimgpoints,
    image,
    translationvectors,
    rotationvectors,
    cameramatrix,
    distcoefficients,
):
    eprint("FindLandmark.py: Attempting to save picture with detection boxes on")
    for cnt in corners:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image, f"{w} x {h}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
    for cnt in rejectedimgpoints:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    for tvec, rvec in zip(translationvectors, rotationvectors):
        cv2.drawFrameAxes(image, cameramatrix, distcoefficients, rvec, tvec, 0.1)

    dt = datetime.datetime.now()
    cv2.imwrite(f"Test123{dt.strftime('%M%S')}.jpeg", image)
    eprint(f"outputted to Test123{dt.strftime('%M%S')}.jpeg")

perform_Findlandmark(initCamera())
