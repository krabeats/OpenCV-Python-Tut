# this needs some of the calibration data

import numpy as np
import cv2 as cv
import glob

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

# draw a cube 
def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    #draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img


criteria = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# axis points for draw_cube()
axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3], [0,3,-3], [3,3,-3], [3,0,-3]])


# axis points for draw()
# axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

#arrays to store object points and image points from all the images
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane 

# added the below from cam_calibration as required (i didnt save this data as an npz file )
for fname in glob.glob('left*.jpg'):
    img_2 = cv.imread(fname)
    gray_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
    ret, corners_2 = cv.findChessboardCorners(gray_2, (7,6), None)
    if ret == True:
        objpoints.append(objp)

        corners2_2 = cv.cornerSubPix(gray_2,corners_2, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2_2)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray_2.shape[::-1], None, None)


for fname in glob.glob('left*.jpg'):
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)

    if ret == True:

        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1), criteria)

        # find the rotation translation vectors
        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

        # project 3d points to img plane
        imgpts, jac = cv.projectPoints(axis, rvecs,tvecs,mtx,dist)

        img = draw_cube(img,corners2,imgpts)
        cv.imshow('img',img)
        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite(fname[:6]+'.png', img)

cv.destroyAllWindows