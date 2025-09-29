import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('hm2.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp = sift.detect(gray,None)

#img=cv.drawKeypoints(gray,kp,img)

#more detailed keypoints
img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('rslt',img)
k = cv.waitKey(0)