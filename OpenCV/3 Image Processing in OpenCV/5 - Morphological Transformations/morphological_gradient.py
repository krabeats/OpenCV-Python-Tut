import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('j.png', cv.IMREAD_GRAYSCALE)
assert img is not None, 'file could not be read, check with os.path.exists()'

# this kinda makes an outline
# https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html#autotoc_md1420

kernel = np.ones((5,5), np.uint16)
gradient = cv.morphologyEx(img,cv.MORPH_GRADIENT,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(gradient),plt.title('Gradient')
plt.xticks([]), plt.yticks([])
plt.show()