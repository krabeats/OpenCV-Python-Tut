import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#roi is the object or region of object we need to find
roi = cv.imread('messi3.jpg')
assert roi is not None, 'check file with os.path.exists()'
hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)

#target is the image we search in
target = cv.imread('messi3.jpg')
assert target is not None, 'check file with os.path.exists()'
hsvt = cv.cvtColor(target, cv.COLOR_BGR2HSV)

#find the histograms using calchist. Can be done with np.histogram2d also
M = cv.calcHist([hsv],[0,1], None, [180,256], [0,180,0,256])
I = cv.calcHist([hsvt],[0,1], None, [180,256], [0,180,0,256])

# got lost below 
# https://docs.opencv.org/4.x/dc/df6/tutorial_py_histogram_backprojection.html
#https://opencvpython.blogspot.com/2013/03/histograms-4-back-projection.html
# it seems belpw is how its done using numpy (the numpy way)

R = M/(I+1)

h,s,v = cv.split(hsvt)
B = R[h.ravel(),s.ravel()]
B = np.minimum(B,1)
B = B.reshape(hsvt.shape[:2])

disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
cv.filter2D(B,-1,disc,B)
B = np.uint8(B)
cv.normalize(B,B,0,255,cv.NORM_MINMAX)

ret,thresh = cv.threshold(B,50,255,0)





