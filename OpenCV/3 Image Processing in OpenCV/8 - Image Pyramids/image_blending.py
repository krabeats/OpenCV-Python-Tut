import cv2 as cv
import numpy as np,sys
 
# for this to work images need to be the same size

A = cv.imread('apple.jpg')
B = cv.imread('orange.jpg')
assert A is not None, "file could not be read, check with os.path.exists()"
assert B is not None, "file could not be read, check with os.path.exists()"

#generate gaussian pyramid for A / gpA = gaussian pyramid A
G = A.copy()
gpA = [G] # use ints to access ary els e.g gpA[i]
for i in range(6):
    G = cv.pyrDown(G)
    gpA.append(G)

#generate gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpB.append(G)

#generate a laplacian pyramid for A / lpA = laplacian pyramid A
lpA = [gpA[5]]
for i in range(5,0,-1): # backwards loop. start from 5 -> 0 stepping back by -1 (stops at 0 and has 5 els only 5,4,3,2,1)
    GE = cv.pyrUp(gpA[i])
    L = cv.subtract(gpA[i-1],GE) # subtract creates a difference between images
    lpA.append(L)

#generate a laplacian pyramid for B
lpB = [gpB[5]]
for i in range(5,0,-1):
    GE = cv.pyrUp(gpB[i])
    L = cv.subtract(gpB[i-1],GE)
    lpB.append(L)


# now add left and right halves of images in each level
LS = []
for la,lb in zip(lpA,lpB): # zip function, check notes
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
    LS.append(ls)

# now reconstruct
ls_ = LS[0]
for i in range(1,6):
    ls_ = cv.pyrUp(ls_)
    ls_ = cv.add(ls_, LS[i])


# image with direct connecting each half
real = np.hstack((A[:,:cols//2],B[:,cols//2:]))

#cv.imshow('display_window', lpA[5])
#k = cv.waitKey(0)

#cv.imwrite('Pyramid_blending2.jpg', ls_)
#cv.imwrite('Direct_blending.jpg', real)