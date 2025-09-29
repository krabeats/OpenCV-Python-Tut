import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('j.png', cv.IMREAD_GRAYSCALE)
assert img is not None, 'file could not be read, check with os.path.exists()'

kernel = np.ones((5,5), np.uint8)
dialation = cv.dilate(img,kernel,iterations = 1)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dialation),plt.title('Dialation')
plt.xticks([]), plt.yticks([])
plt.show()