# based on:
# https://www.geeksforgeeks.org/python/measure-size-of-an-object-using-python-opencv/
#
# see also:
# https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html


import sys

import cv2
import numpy as np

if len(sys.argv) < 2:
    print(sys.argv[0], "<image-file>")
    raise Exception

img = cv2.imread(sys.argv[1])

if img is None:
    raise Exception

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a threshold to the image to
# separate the objects from the background
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find the contours of the objects in the image
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Loop through the contours and calculate the area of each object
for cnt in contours:
    area = cv2.contourArea(cnt)

    # Draw a bounding box around each
    # object and display the area on the image
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, f"{w} x {h}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Show the final image with the bounding boxes
# and areas of the objects overlaid on top
cv2.imshow("image", img)
while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
