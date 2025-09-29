import cv2 as cv
import numpy as np

im = cv.imread('messi2.jpg')
assert im is not None, 'file could not be read, check with os.path.exists()'


imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY) # turns img black n white
ret, thresh = cv.threshold(imgray, 127,255,0) # thresh creates a threshhold, not sure what ret does
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # breakdown of params - https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html

draw_contours = cv.drawContours(im, contours, -1, (0,255,255), 3) # -1 draws all contours, otherwise use a number for the contour you want


cv.imshow('img', draw_contours)
k = cv.waitKey(0)