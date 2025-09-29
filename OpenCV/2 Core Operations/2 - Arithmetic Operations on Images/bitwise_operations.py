import numpy as np
import cv2 as cv

# load two images
img1 = cv.imread('messi5.jpg')
img2 = cv.imread('opencv_logo.jpg')
assert img1 is not None, "file could not be read, check with os.path.exists()"
assert img2 is not None, "file could not be read, check with os.path.exists()"

# i want to put logo on top-left corner, so i create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols]

# now create a mask of logo and create its inverse mask also
img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)

# now black out ths area of the logo in ROI
img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)

# take only region of logo from logo image
img2_fg = cv.bitwise_and(img2,img2,mask = mask)

# put logo in roi and modify the main image
dst = cv.add(img1_bg,img2_fg)
img1[0:rows, 0:cols] = dst

cv.imshow('res',img1)
cv.waitKey(0)
cv.destroyAllWindows()