# Brute-Force Matching with SIFT Descriptors and Ratio Test

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread('main.png', cv.IMREAD_GRAYSCALE)    # query image 
img2 = cv.imread('febreze.png', cv.IMREAD_GRAYSCALE) # train image

# initiate sift detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

#bf matcher with default params 
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

#apply ratio and test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

#cv.drawMatchesKnn expects list of lists as matches
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3),plt.show()