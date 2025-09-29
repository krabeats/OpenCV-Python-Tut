import numpy as np
import cv2 as cv

img = cv.imread('star1.jpg')
assert img is not None, 'check path with os.path.exists()'
img_grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

ret,thresh = cv.threshold(img_grey,127,255,0)
contours,hierarchy = cv.findContours(thresh,2,1)

cnt = contours[0]

# convexity defects
# this is to do with deviations of the object from the hull
hull = cv.convexHull(cnt,returnPoints = False) # false is required to find the defects
defects = cv.convexityDefects(cnt,hull)

print(defects)

for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv.line(img,start,end,[0,255,0],2)
    cv.circle(img,far,5,[0,255,0],-1)

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows