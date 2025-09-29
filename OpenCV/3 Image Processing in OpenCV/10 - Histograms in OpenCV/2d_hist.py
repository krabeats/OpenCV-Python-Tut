import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('messi3.jpg')
assert img is not None, 'check os.path.exists()'
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)


# channels = [0,1] because we need to process both H and S plane.
# bins = [180,256] 180 for H plane and 256 for S plane.
# range = [0,180,0,256] Hue value lies between 0 and 180 & Saturation lies between 0 and 256.
hist = cv.calcHist([hsv], [0,1], None, [180,256], [0,180,0,256])

#cv.imshow('hist',hist)
#k = cv.waitKey(0)

plt.imshow(hist,interpolation= 'nearest')
plt.show()

