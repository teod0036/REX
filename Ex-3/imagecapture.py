# This script shows how to open a camera the picamera2 module and grab frames and show these.
# Kim S. Pedersen, 2023

import cv2 # Import the OpenCV library
import time
from pprint import *
import datetime 


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
''''''
picam2_config = cam.create_still_configuration({"size": imageSize, "format": 'RGB888'},
                                                            controls={"ScalerCrop": (0,0,3280,2464)},
                                                            queue=False)
cam.configure(picam2_config) # Not really necessary
cam.start(show_preview=False)

pprint(cam.camera_configuration()) # Print the camera configuration in use

time.sleep(1)  # wait for camera to setup


# Open a window
#WIN_RF = "Example 1"
#cv2.namedWindow(WIN_RF)
#cv2.moveWindow(WIN_RF, 100, 100)

dt = datetime.datetime.now()
image = cam.capture_file(f"{dt.strftime('%M%S')}.jpeg")

# Show frames
#cv2.imshow(WIN_RF, image)
    

# Finished successfully
