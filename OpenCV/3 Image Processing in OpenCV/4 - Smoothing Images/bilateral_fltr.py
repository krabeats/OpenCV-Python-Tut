import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('woods.jpg')
assert img is not None, 'file could not be read, check with os.path.exists()'

# this type of filter works to preserve corners/lines
blur = cv.bilateralFilter(img,9,75,75)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blur')
plt.xticks([]), plt.yticks([])
plt.show()