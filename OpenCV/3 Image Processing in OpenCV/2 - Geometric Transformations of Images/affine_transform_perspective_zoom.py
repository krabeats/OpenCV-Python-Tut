import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('suduko.png')
assert img is not None, "file could not be read, check with os.path.exists()"
rows,cols,ch = img.shape

pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]]) #[top line points: [x,y],[x,y] / bottom line points: [x,y],[x,y]]
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]]) #use pts2 to adjust where the points should be after - see exmpl imgin folder

M = cv.getPerspectiveTransform(pts1,pts2) #this preforms the perspective transformation

dst = cv.warpPerspective(img,M,(200,200))#this outputs it along with the zoom so you can see it

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()