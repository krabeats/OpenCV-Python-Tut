# brute force matching with orb discriptors

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread('main.png', cv.IMREAD_GRAYSCALE)    # query image 
img2 = cv.imread('febreze.png', cv.IMREAD_GRAYSCALE) # train image

# initiate orb detector
orb = cv.ORB_create()

# find keypoints and descriptors with orb
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create a (brute force) BF matcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# match descriptors 
matches = bf.match(des1,des2)

# sort them in order of their distance 
matches = sorted(matches, key= lambda x:x.distance)

# draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3),plt.show()