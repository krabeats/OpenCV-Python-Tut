import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('j.png', cv.IMREAD_GRAYSCALE)
assert img is not None, 'file could not be read, check with os.path.exists()'

# difference between opening and input of the img

kernel = np.ones((9,9), np.uint8)
tophat = cv.morphologyEx(img,cv.MORPH_TOPHAT,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(tophat),plt.title('tophat')
plt.xticks([]), plt.yticks([])
plt.show()