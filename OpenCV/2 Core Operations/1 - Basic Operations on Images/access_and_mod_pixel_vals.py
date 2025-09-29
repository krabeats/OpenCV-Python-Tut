import numpy as np
import cv2 as cv

img = cv.imread('messi5.jpg')
assert img is not None, "File could not be read, check with os.path.exists()"

px = img[100,100]
print (px)

# accessing only blue pixel
blue = img[100,100,0]
print (blue)

# you can modify the px values the same way
img[100,100] = [255,255,255]
print(img[100,100])

# accessing image properties 
# these inc: rows, collumns and channels (e.g: bgr)

# the shape of an img returns a tuple of rows, collumns and channels
print('image shape  = ', img.shape)

# size = total px in an image
print(img.size)

# datatype 
print(img.dtype)
# note 
# img.dtype is very important while debugging because a large number of errors in OpenCV-Python code are caused by invalid datatype.



