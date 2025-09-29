import numpy as np
import cv2 as cv

img = cv.imread('cv1.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, 'check os.path.exists()'
img = cv.medianBlur(img,5)
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)

# the circle params 1 and 2 need to be adjusted to limit how many circles are found. 
# check out function discription for param definitions
circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
                          param1=50,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the centre of the circle
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv.imshow('detected circles',cimg)
cv.waitKey(0)
cv.destroyAllWindows()