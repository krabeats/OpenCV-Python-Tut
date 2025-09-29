# For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255]. Different software use different scales.

import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while(1):

    # take each frame
    _, frame = cap.read()

    # convrt bgr to hsv
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #define range of blue color in hsv
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    #threshold the hsv image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    # bitwise-and mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)

    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows