import numpy as np
import cv2 as cv

img = cv.imread('star.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None,'file could not be read, use os.path.exists()'

ret,thresh = cv.threshold(img,127,255,0)
contours,hierarchy = cv.findContours(thresh,1,2)

cnt = contours[0]


#get the aspect ratio of bounding rectangle of the object
x,y,w,h = cv.boundingRect
aspect_ratio = float(w)/h


#get the extent, ration of contour area to bounding rect area
area = cv.contourArea(cnt)
x,y,w,h = cv.boundingRect(cnt)
rect_area = w*h
extent = float(area)/rect_area


#soliditys is the ratio of contour area to its convex hull
area = cv.contourArea(cnt)
hull = cv.convexHull(cnt)
hull_area = cv.contourArea(hull)
solidity = float(area)/hull_area


#equivilant diameter is the diameter of the circle whose area is the same as the contour area
area = cv.contourArea(cnt)
equi_diameter = np.sqrt(4*area/np.pi)


#orientation is the angle at which the object is directed - also gives major/minor axis lengths
(x,y),(MA,ma),angle = cv.fitEllipse(cnt)


#mask and pixel points 
mask = np.zeros(img.shape,np.uint8)
cv.drawContours(mask,[cnt],0,255,-1)
pixelpoints = np.transpose(np.nonzero(mask)) # coordinates provided in (row, col)
pixelpoints = cv.findNonZero(mask) # coordinates provided in (x,y)


# max val, min val, and their locations
min_val, max_val, min_loc, = cv.minMaxLoc(img,mask = mask)


#mean colour or intensity
mask = np.zeros(img.shape,np.uint8)
mean_val = cv.mean(img,mask = mask)


#extreme points is the top, bottom, left, right most points of an object
# reference extreme points img
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0]) # min
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0]) # max
topmost = tuple(cnt[cnt[:,:,1].argmin()][0]) # min
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0]) # max