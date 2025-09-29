import numpy as np
import cv2 as cv

img = cv.imread('blck2.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)

# result is dilated for for marking corners, not important
dst = cv.dilate(dst,None)

# threshold for optimum value, it may vary depending on the img
img[dst>0.01*dst.max()] = [0,0,255]

cv.imshow('dst', img)
if cv.waitKey(0) == 0xff == 27:
    cv.destroyAllWindows()