# Example showing how to grab frames using the PiCamera module instead of OpenCV
# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from fractions import *
import time
import cv2
 
 
print("OpenCV version = " + cv2.__version__)
 
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
time.sleep(1) # Wait for camera

camera.resolution = (640, 480)
camera.framerate = 30

camera.shutter_speed = camera.exposure_speed
camera.exposure_mode = 'off'

gain = camera.awb_gains
camera.awb_mode='off'
#gain = (Fraction(2,1), Fraction(1,1))
#gain = (1.5, 1.5)
camera.awb_gains = gain

print("shutter_speed = ", camera.shutter_speed)
print("awb_gains = ", gain)

rawCapture = PiRGBArray(camera, size=camera.resolution)
 
# Open a window
WIN_RF = "Frame";
cv2.namedWindow(WIN_RF);
cv2.moveWindow(WIN_RF, 100       , 100);


# allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image
	image = frame.array
 
	# show the frame
	cv2.imshow(WIN_RF, image)
	key = cv2.waitKey(4) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break


		
