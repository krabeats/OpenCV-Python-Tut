import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('jnoise2.png', cv.IMREAD_GRAYSCALE)
assert img is not None, 'file could not be read, check with os.path.exists()'

# closing is usefull for removing noise on the inside 

kernel = np.ones((5,5), np.uint8)
closing = cv.morphologyEx(img,cv.MORPH_CLOSE,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(closing),plt.title('Closing')
plt.xticks([]), plt.yticks([])
plt.show()