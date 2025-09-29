import numpy as np
import cv2 as cv

img = cv.imread('messi5.jpg')
assert img is not None, "File could not be read, check with os.path.exists()"

#split - this takes time so only use when necessary
#b,g,r = cv.split(img)

# merge 
#img = cv.merge((b,g,r))

#set red px to 0
img[:,:,2] = 0

cv.imshow("Display window", img)
k = cv.waitKey(0)

