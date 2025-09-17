import datetime
import time
from pprint import pprint

import cv2  # Import the OpenCV library
import numpy as np


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


def takePictureOnly():
    try:
        import picamera2

        print("Camera.py: Using picamera2 module")
    except ImportError:
        print("Camera.py: picamera2 module not available")
        exit(-1)

    print("OpenCV version = " + cv2.__version__)

    # Open a camera device for capturing
    imageSize = (1640, 1232)
    cam = picamera2.Picamera2()
    # Change configuration to set resolution, framerate
    """"""
    picam2_config = cam.create_still_configuration(
        {"size": imageSize, "format": "RGB888"},
        controls={"ScalerCrop": (0, 0, 3280, 2464)},
        queue=False,
    )
    cam.configure(picam2_config)  # Not really necessary
    cam.start(show_preview=False)

    pprint(cam.camera_configuration())  # Print the camera configuration in use
    time.sleep(1)  # wait for camera to setup

    array = cam.capture_array("main")

    return array


def perform_Findlandmark():
    print("FindLandmark.py: Taking a picture using imagecapture")

    image = takePictureOnly()

    print("FindLandmark.py: Fetching the dictionary")
    dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    print("FindLandmark.py: Attempting to detect Markers")
    corners, ids, rejectedimgpoints = cv2.aruco.detectMarkers(image, dict)
    for cnt in corners:
        print("printing corner" + str(cnt))

    if ids is None:
        print("FindLandmark.py: No landmarks found, ending FindLandmark")
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

    CreateDetectionImage(
        corners,
        rejectedimgpoints,
        image,
        translationvectors,
        rotationvectors,
        cameramatrix,
        distcoefficients,
    )

    return objpoints


def CreateDetectionImage(
    corners,
    rejectedimgpoints,
    image,
    translationvectors,
    rotationvectors,
    cameramatrix,
    distcoefficients,
):
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

    for tvec, rvec in zip(translationvectors, rotationvectors):
        cv2.drawFrameAxes(image, cameramatrix, distcoefficients, rvec, tvec, 0.1)

    dt = datetime.datetime.now()
    cv2.imwrite(f"Test123{dt.strftime('%M%S')}.jpeg", image)
    print(f"outputted to Test123{dt.strftime('%M%S')}.jpeg")


print(perform_Findlandmark())
