import numpy as np
import cv2 as cv

img = cv.imread('home.jpg')
Z = img.reshape((-1,3))

#convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(k) and apply kmeans 
criteria = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER,10,1.0)
# k = equals the quantize number (number of colours)
K = 8
ret,label,center = cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

# now convert back into uint8, and make original image 
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv.imshow('res2',res2)
cv.waitKey(0)
cv.destroyAllWindows()