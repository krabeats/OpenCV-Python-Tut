import numpy as np
import cv2 as cv

img = cv.imread('messi5.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"

# inter_area for shrinking
# inter_cubic / inter_linear for scaling [linear is the default]


res = cv.resize(img,None,fx=0.5,fy=0.5, interpolation=cv.INTER_AREA)

# or 

heigt, width = img.shape[:2]
res2 = cv.resize(img,(2*width, 2*heigt), interpolation=cv.INTER_CUBIC)

cv.imshow('result',res)
cv.waitKey(0)
cv.destroyAllWindows