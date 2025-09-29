import numpy as np
import cv2 as cv

green = np.uint8([[[0,255,0]]])
hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
print(hsv_green)
# result =  [[[60 255 255]]]

# to get the upper and lower -10 or +10 from first array number (eg 60)
# you can also use gimp tut says
