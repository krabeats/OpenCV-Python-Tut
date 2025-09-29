import numpy as np
import cv2 as cv

img = cv.imread('messi5.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
rows,cols = img.shape

#cols-1 and row-1 are the coordinate limits
M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1) # ((position related), rotation, img scale)
dst = cv.warpAffine(img,M,(cols,rows))

cv.imshow('result',dst)
cv.waitKey(0)
cv.destroyAllWindows