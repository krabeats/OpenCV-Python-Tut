import numpy as np
import cv2 as cv

# https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html

# create a black image
img = np.zeros((512,512,3), np.uint8)

# draw a diagonal blue line with thickness of 5 px
cv.line(img,(0,0),(511,511),(255,0,0),5)

# all shapes have specific params: ie: colour, position, size, line thickness/fill etc
cv.rectangle(img,(384,0),(510,128),(0,255,0),3)

cv.circle(img,(447,63), 63, (0,0,255), -1)

cv.ellipse(img,(256,256),(100,50),0,0,180,255,-1)

#drawing a polygon
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(img,[pts],True,(0,255,255))

# adding text to an img
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)

cv.imshow("Display window", img)
k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("draw.png", img)