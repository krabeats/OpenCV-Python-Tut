import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('jnoise.png', cv.IMREAD_GRAYSCALE)
assert img is not None, 'file could not be read, check with os.path.exists()'

# opening is usefull for removing noise on the outside 

kernel = np.ones((5,5), np.uint8)
opening = cv.morphologyEx(img,cv.MORPH_OPEN,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(opening),plt.title('Opening')
plt.xticks([]), plt.yticks([])
plt.show()