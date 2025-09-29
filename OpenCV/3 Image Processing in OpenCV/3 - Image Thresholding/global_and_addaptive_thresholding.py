import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# when doing threshold work the image must be converted to grayscale as seen below
img = cv.imread('messi5.jpg', cv.IMREAD_GRAYSCALE) # suduko.png
assert img is not None, 'file could not be read, check with os.path.exists()'
img = cv.medianBlur(img,5)

ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY) # the back slash is for a new line. its not required
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
                           cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                           cv.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
           'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'grey') #subplot layout = 2 by 2 and 1 = img / grey is the name of the img but doesnt show
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([]) # removes the numbers from x and y axis
plt.show()