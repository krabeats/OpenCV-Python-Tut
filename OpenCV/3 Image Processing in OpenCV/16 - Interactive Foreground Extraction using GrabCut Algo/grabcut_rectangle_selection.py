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

plt.imshow(img),plt.colorbar(),plt.show()

