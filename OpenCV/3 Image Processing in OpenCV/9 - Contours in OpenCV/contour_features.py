import numpy as np
import cv2 as cv

img = cv.imread('star.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None,'file could not be read, use os.path.exists()'
ret,thresh = cv.threshold(img,127,255,0)
contours,hierarchy = cv.findContours(thresh,1,2)

cnt = contours[0]
M = cv.moments(cnt)
#print(M)

# this how you calculate the centroid
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

#print(cy)

area = cv.contourArea(cnt)
#print(area)

# adding true means the shape is closed. if not, shape is just a curve
perimiter = cv.arcLength(cnt,True)



# contour approximation
# #https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html#autotoc_md1315
# check approx img for reference
epsilon = 0.1*cv.arcLength(cnt,True)
approx = cv.approxPolyDP(cnt,epsilon,True)

# convex hull
# https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html#autotoc_md1316
# check convex hull image for reference
#
# hull = cv.convexHull(points[, hull[, clockwise[, returnPoints]]])
#Arguments details:
#points are the contours we pass into.
#hull is the output, normally we avoid it.
#clockwise : Orientation flag. If it is True, the output convex hull is oriented clockwise. Otherwise, it is oriented counter-clockwise.
#returnPoints : By default, True. Then it returns the coordinates of the hull points. If False, it returns the indices of contour points corresponding to the hull points.
#So to get a convex hull as in above image, following is sufficient:
hull = cv.convexHull(cnt)
#But if you want to find convexity defects, you need to pass returnPoints = False.

#check if a curve is convex or not - returns true or false
convex_check = cv.isContourConvex(cnt)
#print(convex_check)

# bounding rectangles 
# staight bounded rectangle - draws a square/rectange around img normally
# check bounding rec for reference
x,y,w,h = cv.boundingRect(cnt)
cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# rotated rectangle - draws a square/rectangle with minimum area (hence rotated)
rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int0(box)
cv.drawContours(img,[box],0,(0,0,255),2)


# minimum enclosed circle - reference circumcircle img
(x,y),radius = cv.minEnclosingCircle(cnt)
centre = (int(x),int(y))
radius = int(radius)
cv.circle(img,centre,radius,(0,255,0),2)


# fitting an ellipse - reference fitellipse img
ellipse = cv.fitEllipse(cnt)
cv.ellipse(img,ellipse,(0,255,0),2)

#fitting a line - reference fitline
rows,cols = img.shape[:2]
[vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0,01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
