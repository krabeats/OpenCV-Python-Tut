import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('messi3.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, 'use os.path.exists()'

# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255 #[x,y] x = length / y = hieght
masked_img = cv.bitwise_and(img,img,mask = mask)

# calc histogram with mask and without

# check 3rd arg for mask
hist_full = cv.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv.calcHist([img],[0],mask,[256],[0,256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray') 
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])

plt.show()