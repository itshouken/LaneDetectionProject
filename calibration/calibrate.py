import numpy as np
import cv2
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0, 0, 0), (1, 0, 0), ... ,(6, 5, 0)
objp = np.zeros((6*8, 3), np.float32)
objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2) * 30

# Arrays to store object points and image points 
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

images = glob.glob('*.png')

for fname in images:
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	# Find the chess board corners
	ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
	
	# If found, add object points, image points
	if ret == True:
		objpoints.append(objp)
		corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
		imgpoints.append(corners2)

		# Draw and display the corners
		img = cv2.drawChessboardCorners(img, (8,6), corners2, ret)
		cv2.imshow('img', img)
		cv2.waitKey(10)
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv2.imread('left-0000.png')
h, w = img.shape[:2]

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

x,y,w,h = roi

#cv2.imshow('calibrate',dst)
#cv2.waitKey(ord('q'))
