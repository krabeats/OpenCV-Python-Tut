import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('opencv_logo.png')
assert img is not None, 'file could not be read, check with os.path.exists()'

# blur = cv.blur(img,(5,5))

# gaussin kernel grid min is 3x3
# blur = cv.GaussianBlur(img,(3,3),0)

# median needs a positive odd int
# this is best used on noisy colour
median = cv.medianBlur(img,5)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(blur),plt.title('Blur')
# plt.subplot(122),plt.imshow(blur),plt.title('Gaussian')
plt.subplot(122),plt.imshow(median),plt.title('Median')
plt.xticks([]), plt.yticks([])
plt.show()