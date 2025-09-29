import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('messi5.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, 'check os.path.exists(img)'

dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# -- below creates the inverse


rows, cols = img.shape
crow, ccol = rows//2, cols//2

#create a mask first, centre square is 1, remaining all zeros
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1 # increase the 30 to reduce the filtering of high frequency

#apply mask and invert DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
image_back = cv.idft(f_ishift)
image_back = cv.magnitude(image_back[:,:,0],image_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Image Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(image_back, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
 
plt.show()