import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('messi3.jpg')
assert img is not None, 'check os.path.exists()'
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
h,s,v = cv.split(hsv)

# first argument is h plane, second is s plane (hue / saturation i believe)
hist, xbins, ybins = np.histogram2d(h.ravel(),s.ravel(),[180,256],[[0,180],[0,256]])

