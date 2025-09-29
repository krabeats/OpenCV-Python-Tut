import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('j.png', cv.IMREAD_GRAYSCALE)
assert img is not None, 'file could not be read, check with os.path.exists()'

# difference between closingof the input img and the input img

kernel = np.ones((9,9), np.uint8)
blackhat = cv.morphologyEx(img,cv.MORPH_BLACKHAT,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blackhat),plt.title('blackhat')
plt.xticks([]), plt.yticks([])
plt.show()