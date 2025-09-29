import cv2 as cv
import numpy as np

img = cv.imread('messi3.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, 'check file with os.path.exists()'

equ = cv.equalizeHist(img)
res = np.hstack((img,equ)) # stack imgs side by side 
cv.imshow('img', res)
k = cv.waitKey(0)