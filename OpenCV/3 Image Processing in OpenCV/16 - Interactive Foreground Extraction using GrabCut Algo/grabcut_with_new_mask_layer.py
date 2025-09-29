import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
im = cv.imread('messi5.jpg')
assert im is not None, "file could not be read, check with os.path.exists()"
img = cv.cvtColor(im,cv.COLOR_BGR2RGB)
mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (70,188,1088,920) # (x,y, - x,y)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

# adding the new mask as shown in the opencv tut has caused a bit of a problem 
# alot of tweaking will be needed to get the image cropped right
# https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html

# new mask is the mask img manually labelled/created
newmask = cv.imread('mask.jpg', cv.IMREAD_GRAYSCALE)
assert newmask is not None, "file could not be read, check with os.path.exists()"

# wherever it is marked white (sure foreground), change mask=1
# wherever it is marked black (sure background), change mask=0
mask[newmask == 0] = 0
mask[newmask == 255] = 1

mask, bgdModel, fgdModel = cv.grabCut(img,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)

mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()