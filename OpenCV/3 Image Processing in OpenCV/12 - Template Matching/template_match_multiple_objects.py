import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv.imread('mario.jpg')
assert img_rgb is not None, 'try os.path.exists() for img'
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('coin.jpg', cv.IMREAD_GRAYSCALE)
assert template is not None, 'try os.path.exists() for template'

w,h = template.shape[::-1]

res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1) # last param is line thickness and must be an int

cv.imshow('res', img_rgb)
k = cv.waitKey(0)
