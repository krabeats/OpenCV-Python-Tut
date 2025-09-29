import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('messi3.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, 'file not found, check with os.path.exists()'

# https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html
#above gives breakdown of all params 
hist = cv.calcHist([img],[0],None,[256],[0,256])

#numpy histogram func
# hist,bins = np.histogram(img.ravel(),256,[0,256])

im2 = cv.imread('messi3.jpg')
rgb = cv.cvtColor(im2, cv.COLOR_BGR2RGB)

# matplot has a histogram plotting func
plt.subplot(1,2,1), plt.hist(img.ravel(),256,[0,256])
plt.subplot(1,2,2),plt.imshow(rgb)
plt.show()
