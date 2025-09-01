# this is an example showing how ArUco markers are detected with OpenCV APIs. Note this just covers some essential steps for an illustration purpose.
# the steps include thresholding, finding contours and locating rectangular corner points. the pose calculation is not included since the camera matrix for taking the sample pic is unknown.
# tested under opencv 4.8.0
import cv2

img = cv2.imread("singlemarkersoriginal.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# cv2.imshow("marker", gray)

# cv2.waitKey(0)

th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 6)

# cv2.imshow("marker", th)

# cv2.waitKey(0)

ker = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
img2 = cv2.morphologyEx(th, cv2.MORPH_OPEN, ker)

contours, hierarchy = cv2.findContours(img2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#markers_candidate = contours
markers_candidate = []

#filter out unlikely contours
for i in range(len(contours)):
    #can this be approximated as a convex shape with four points?
    approxCurve = cv2.approxPolyDP(contours[i], len(contours[i]) * 0.05, True)
    #ensure points distance are large enough
    if len(approxCurve) == 4 and cv2.isContourConvex(approxCurve):

        dists = [cv2.norm(approxCurve[j] - approxCurve[(j+1)%4]) for j in range(4)]
        if min(dists) < 15 or max(dists) > 80:
            continue
            
        markers_candidate.append(approxCurve)



img3 = cv2.drawContours(img, markers_candidate, -1, (0,255,0), 1)

cv2.imshow("marker", img3)

cv2.waitKey(0)