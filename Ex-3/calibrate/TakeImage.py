
import cv2 # Import the OpenCV library
import time
from pprint import pprint
import datetime 


def takePicture():
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
    picam2_config = cam.create_still_configuration({"size": imageSize, "format": 'RGB888'},
                                                                controls={"ScalerCrop": (0,0,3280,2464)},
                                                                queue=False)
    cam.configure(picam2_config) # Not really necessary
    cam.start(show_preview=False)

    pprint(cam.camera_configuration()) # Print the camera configuration in use

    time.sleep(1)  # wait for camera to setup

    dt = datetime.datetime.now()
    array = cam.capture_array("main")

    return array
