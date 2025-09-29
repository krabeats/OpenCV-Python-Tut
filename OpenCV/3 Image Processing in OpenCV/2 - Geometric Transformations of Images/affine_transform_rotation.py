import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('suduko.png')
assert img is not None, "file could not be read, check with os.path.exists()"
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]]) #[x,y],[x,y],[x,y]
pts2 = np.float32([[10,100],[200,50],[100,250]]) #[x,y],[x,y],[x,y]

M = cv.getAffineTransform(pts1,pts2) #this preforms the transformation

dst = cv.warpAffine(img,M,(cols,rows))#this outputs it so you can see it

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()