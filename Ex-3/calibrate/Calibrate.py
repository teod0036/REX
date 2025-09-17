import datetime

import cv2 as cv
import numpy as np

from TakeImage import takePicture

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

chessboardSize = 5, 4

img = takePicture()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

print("taking image...")

# Find the chess board corners
ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

# If found, add object points, image points (after refining them)
if ret == True:
    print("found!")

    objpoints.append(objp)

    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
    dt = datetime.datetime.now()
    cv.imwrite(f"Raw{dt.strftime('%M%S')}.jpeg", img)

    print(f"saving to {dt.strftime('%M%S')}.jpeg")

cv.destroyAllWindows()
