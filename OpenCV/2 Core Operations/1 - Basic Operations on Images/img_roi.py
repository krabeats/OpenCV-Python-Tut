import numpy as np
import cv2 as cv

img = cv.imread('messi5.jpg')
assert img is not None, "File could not be read, check with os.path.exists()"

# original from tut but my picture is bigger
#ball = img[280:340, 330:390] #[y1:y1, x1:x2]
#img[273:333, 100:160] = ball 

ball = img[775:910, 750:900] #[y1:y1, x1:x2]
img[775:910, 90:240] = ball 
# if you want to move a section left/right you only need to update the x axis
# for up/down the y axis

# to make it easy, find the point (x,y) and then add to the x and the y the
# amount you need down for the y and right for the x. This will let you move 
# around with the right proportions required to make this work

cv.imshow("Display window", img)
k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("messi5_updated.jpg", img) # saved as jpg as png to big