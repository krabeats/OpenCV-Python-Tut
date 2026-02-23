import numpy as np
import cv2 as cv

img = cv.imread('messi_paint_lines.jpg')
mask = cv.imread('mask.png', cv.IMREAD_GRAYSCALE)

#you can use either inpaint_ns or _telea
#it  might be worth experimenting in small sections 
#to get a better result. 
dst = cv.inpaint(img,mask,3,cv.INPAINT_TELEA)


cv.imshow('dst',dst)
cv.waitKey(0)
cv.destroyAllWindows()