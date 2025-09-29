import numpy as np
import cv2 as cv

img = cv.imread('messi5.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
rows,cols = img.shape

M = np.float32([[1,0,100],[0,1,50]]) # [1,0,x] [0,1,y]
dst = cv.warpAffine(img,M,(cols,rows))

cv.imshow('result',dst)
cv.waitKey(0)
cv.destroyAllWindows