import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

x = np.random.randint(25,100,25)
y = np.random.randint(175,255,25)
z = np.hstack((x,y))
z = z.reshape((50,1))
z = np.float32(z)
#plt.hist(z,256,[0,256]),plt.show()

# ---- part 2

# define criteria ( type, max_iter = 10, epsilon = 1.0 )
criteria = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER,10,1.0)

# set flags (just to avoid line break in the code)
flags = cv.KMEANS_RANDOM_CENTERS

# apply kmeans 
compactness,labels,centers = cv.kmeans(z,2,None,criteria,10,flags)

# -- 

A = z[labels==0]
B = z[labels==1]

# now plot A in red, B in blue, and centers in yellow 

plt.hist(A,256,[0,256],color = 'r')
plt.hist(B,256,[0,256],color = 'b')
plt.hist(centers,32,[0,256],color = 'y')
plt.show()