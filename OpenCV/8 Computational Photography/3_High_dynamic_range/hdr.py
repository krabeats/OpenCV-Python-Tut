import numpy as np
import cv2 as cv

# load exposure images into a list
img_fn = ["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg"]
img_list = [cv.imread(fn) for fn in img_fn]

#exposure times must be in float32 and seconds
exposure_times = np.array([15.0, 2.5, 0.25, 0.0333])

# merge exposures to hdr img
merge_debevec = cv.createMergeDebevec()
hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())
merge_robertson = cv.createMergeRobertson()
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())
 
#tonemap hdr images using gamma correction (set gamma=2.2 for standard display brightness)
tonemap1 = cv.createTonemap(gamma=2.2)
res_debevec = tonemap1.process(hdr_debevec.copy())
res_robertson = tonemap1.process(hdr_robertson.copy())

# merge exposures using mertons fusion
# this alternative method to merge the exposure images doesn't need exposure times 
# or a tonemap as it gives us the result in the range [0..1]
#--
merge_mertens = cv.createMergeMertens()
res_mertens = merge_mertens.process(img_list)

# convert to 8 bits and save
# to display the results we need to convert the data 8bit ints in the range of 0..255
#--
# convert datatype to 8bit and save
res_debevec_8bit = np.clip(res_debevec*255,0,255).astype('uint8')
res_robertson_8bit = np.clip(res_robertson*255,0,255).astype('uint8')
res_mertens_8bit = np.clip(res_mertens*255,0,255).astype('uint8')

# seems there is an issue with debevec results when running on windows.
# this is mentioned online
cv.imwrite("ldr_debevec.jpg", res_debevec_8bit)
cv.imwrite("ldr_robertson.jpg", res_robertson_8bit)
cv.imwrite("fusion_mertens.jpg", res_mertens_8bit)