import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('opencv_logo.png')
assert img is not None, 'file could not be read, check with os.path.exists()'

# kernel creates a 5x5 grid around a pixel and gets the average and replaces that 
# pixel. 5x5 is the grid and the 25 is the amount of pixels around the grid use to
# create the average. if you use more than 25 like 50 you then use the same pixels twice
# witch create a double bias average (lighter img in this case) and vise versa if you use
# less.
kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()