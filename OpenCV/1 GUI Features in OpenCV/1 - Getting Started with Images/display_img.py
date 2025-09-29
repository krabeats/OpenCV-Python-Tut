# https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html

import cv2 as cv
import sys

img = cv.imread(cv.samples.findFile("starry_night.jpg"))

if img is None:
    sys.exit("could not read the image.")

cv.imshow("Display window", img)
k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("starry_night.png", img)