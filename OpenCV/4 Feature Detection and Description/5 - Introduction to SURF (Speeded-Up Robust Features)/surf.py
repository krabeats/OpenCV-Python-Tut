# -- Surf is not included in open cv anymore even in opencv-contrib-python (full version)

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('btrfly.jpg', cv.IMREAD_GRAYSCALE)

# create SURF object. You can specify params here or later.
# here i set Hessian threshold to 400
surf = cv.xfeatures2d.SURF_create(400)

# find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img,None)

len(kp)

