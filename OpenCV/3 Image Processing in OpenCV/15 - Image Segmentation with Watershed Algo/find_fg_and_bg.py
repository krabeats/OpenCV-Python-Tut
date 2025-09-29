import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
img = cv.imread('coins.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# noise removal
kernal = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernal, iterations=2)

# sure background area
sure_bg = cv.dilate(opening,kernal,iterations=3)

# finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# finding unknown region
# this is around the coin outline
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)


cv.imshow('thrsh', unknown)
k = cv.waitKey(0)