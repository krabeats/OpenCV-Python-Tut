import cv2 as cv 
import numpy as np

img = cv.imread(cv.samples.findFile('suduko.png')) # su3.png sduko2.jpg
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,150,apertureSize = 3)

# this seems like it only gets the really straight lines. it doesn't if it curves a bit.
lines = cv.HoughLines(edges,1,np.pi/180,200) # param 4 is the threshhold. reduce to get more lines
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv.line(img,(x1,y1),(x2,y2),(0,0,255),2) # thickness is 2 but when its one you can see lines on both sides of a line.

cv.imshow('hough_lines', img)
k = cv.waitKey(0)