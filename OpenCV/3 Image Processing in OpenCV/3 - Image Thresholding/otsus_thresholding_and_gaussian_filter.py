import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
# how to calculate otsus binarization works 

img = cv.imread('noisy.png', cv.IMREAD_GRAYSCALE)
assert img is not None, 'file could not be found, check with os.path.exists()'

# global thresholding
ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)

# otsus threshold
ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# otsus thresholding after gaussian filtering
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# plot all the images
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]

titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
          'Original Noisy Image', 'Histogram', 'Otsus Thresholding',
          'Gaussian filtered Image', 'Histogram', 'Otsus Thresholding']

for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]),plt.xticks([]),plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]),plt.xticks([]),plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()