import numpy as np
import cv2 as cv

img1 = cv.imread('ml.png')
img2 = cv.imread('opencv_logo.png')

assert img1 is not None, "file could not be read, check with os.path.exists()"
assert img2 is not None, "file could not be read, check with os.path.exists()"

dst = cv.addWeighted(img1,0.7,img2,0.3,0)

cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows