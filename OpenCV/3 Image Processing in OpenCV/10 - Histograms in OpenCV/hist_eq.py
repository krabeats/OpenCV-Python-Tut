import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('messi3.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, 'check os.path.exists()'
im1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)

hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

#histogram eq equation
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

# img with eq equation applied 
img2 = cdf[img]

im2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

plt.subplot(131), plt.imshow(im1)
plt.subplot(132), plt.imshow(im2)
#plt.subplot(122), plt.plot(cdf_normalized, color = 'b')
plt.subplot(133), plt.plot(cdf, color = 'b')
plt.hist(img.flatten(),256,[0,256],color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()