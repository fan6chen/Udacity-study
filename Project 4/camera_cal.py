import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pylab

# Read in and make a list of calibration images
images = glob.glob('/home/clement/CarND-Advanced-Lane-Lines/camera_cal/calibration*.jpg')

# Arrays to store object points and image points from all the images
objpoints = [] # 3D points in real world image
imgpoints = [] # 2D points in image plane

# Prepare object points
nx = 9
ny = 6
objp = np.zeros((ny*nx, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)

for idx, fname in enumerate(images):
    # read in each image
    img = mpimg.imread(fname)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If corners are found, add object points, image points
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        # Draw and display the corners
        #img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        #write_name = 'corners_found' + str(idx) + '.jpg'
        #cv2.imwrite(write_name, img)

# load image for reference
img = mpimg.imread('/home/clement/CarND-Advanced-Lane-Lines/camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# test
dst = cv2.undistort(img, mtx, dist, None, mtx)
write_name = './camera_cal/calibration'+str(100)+'.jpg'
cv2.imwrite(write_name, dst)

# Save the camera calibration result for later use
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open("./calibration_pickle.p", "wb"))


