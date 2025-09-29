# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# this usefull when there are over bright areas in an image.
# it stops the image from becoming too bright and loosing detail

import cv2 as cv
import numpy as np

img = cv.imread('messi3.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, 'check os.path.exists()'

# create a clahe object (args are optional)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

res = np.hstack((img,cl1))

cv.imshow('img', res)
k = cv.waitKey(0)