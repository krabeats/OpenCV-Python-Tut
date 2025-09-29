import cv2 as cv

img = cv.imread('messi5.jpg')
assert img is not None, 'file could not be read, check with os.path.exists'

#image pyramids are used to generate the same image in different sizes to detect
#an object at different resolutions 

#pyrdown reduces the size of an image width/legth to half. this can be done x times
lower_reso = cv.pyrDown(img)
lower_reso2 = cv.pyrDown(lower_reso)
lower_reso3 = cv.pyrDown(lower_reso2)
lower_reso4 = cv.pyrDown(lower_reso3)

#pyrup icreases size back at a reduced resolution
higher_reso2 = cv.pyrUp(lower_reso)

#create a laplacian image (this is the black and white version of outlines)
edge = cv.Canny(higher_reso2,100,200)

cv.imshow("Display window", img)
k = cv.waitKey(0)