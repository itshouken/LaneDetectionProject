from __future__ import division
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import pickle
import io
import os
import glob

images = glob.glob('camera_cal/calibration*.jpg')
print(images)

img = mpimg.imread(images[0])
#plt.imshow(img)
#plt.show()
#cv2.waitKey(0)
#plt.close('all')
chess_points = []
image_points = []

chess_point = np.zeros((9*6, 3), np.float32)

chess_point[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

for image in images:
	img = mpimg.imread(image)
	# convert to grayscale
	gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
	
	# returns boolean and coordinates
	success, corners = cv.findChessboardCorners(gray, (9,6), None)
	if success:
		image_points.append(corners)
		chess_points.append(chess_point)
	else:
		print('corners not found {}'.format(image))

image = mpimg.imread('./camera_cal/calibration2.jpg')
#plt.figure()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
#ax1.imshow(image)
#plt.show()
#ax1.set_title('Captured Image', fontsize=30)

gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
ret, corners = cv.findChessboardCorners(gray, (9, 6), None)
if ret == False:
	print('corners not found')
img_original = image.copy()
#test = image.copy()
img1 = cv.drawChessboardCorners(img_original, (9,6), corners, ret)
#plt.imshow(image)
#plt.show()
#ax2.imshow(img1) #img1)
#plt.show()
#ax2.set_title('corners drawn image', fontsize=30)
#plt.tight_layout()
#plt.savefig('saved/chess_corners.png')
#plt.show()


points_pkl = {}
points_pkl["chesspoints"] = chess_points
points_pkl["imagepoints"] = image_points
points_pkl["imagesize"] = (img.shape[1], img.shape[0])
pickle.dump(points_pkl, open("test_data.pkl", "wb"), protocol=2)




# Distortion correction
points_pickle = pickle.load( open( "test_data.pkl", "rb") )
chess_points = points_pickle["chesspoints"]
image_points = points_pickle["imagepoints"]
img_size = points_pickle["imagesize"]

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(chess_points, image_points, img_size, None, None)


# Now save the distortion correction coefficients
camera = {}
camera["mtx"] = mtx
camera["dist"] = dist
camera["imagesize"] = img_size
pickle.dump(camera, open("camera_internal_param.pkl", "wb"), protocol=2)

def distort_correct(img, mtx, dist, camera_img_size):
	img_size1 = (img.shape[1], img.shape[0])
	assert (img_size1 == camera_img_size), 'Image size is not compatible'
	undist = cv.undistort(img, mtx, dist, None, mtx)
	return undist

img = mpimg.imread('./camera_cal/calibration2.jpg')
img_size1 = (img.shape[1], img.shape[0])

undist = distort_correct(img, mtx, dist, img_size1)

# Visualize the captured
plt.figure()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Captured Image', fontsize=30)
ax2.imshow(undist)
ax2.set_title('Undistored Image', fontsize=30)
plt.tight_layout()
plt.savefig('saved_figures/undistored_chess.png')
#plt.show()
# now get an undistored road image
image = mpimg.imread('test_images/test1.jpg')
img_size = (image.shape[1], image.shape[0])
image = distort_correct(image, mtx, dist, img_size)
plt.imshow(image)
#plt.show()
plt.cla()
plt.clf()

def abs_sobel_thresh(img, orient='x', thresh=(0,255)):
	# Convert to grayscale
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	# Apply x or y gradient 
	if orient == 'x':
		abs_sobel = np.absolute(cv.Sobel(gray, cv.CV_64F, 1, 0))
	if orient == 'y':
		abs_sobel = np.absolute(cv.Sobel(gray, cv.CV_64F, 0, 1))
	# Rescale to 8 bit integer
	scaled_sobel = np.uint8(255 * abs_sobel/np.max(abs_sobel))
	# Create a copy and apply the threshold
	binary_output = np.zeros_like(scaled_sobel)
	# Here Iam using inclusive
	binary_output[(scaled_sobel>=thresh[0]) & (scaled_sobel <= thresh[1])] = 1
	return binary_output

plt.imshow(abs_sobel_thresh(image, thresh=(20, 110)), cmap='gray')
#plt.show()
plt.clf()

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
	gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
	x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=sobel_kernel)
	y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=sobel_kernel)

	mag = np.sqrt(x**2 + y **2)
	scale = np.max(mag) / 255
	eightbit = (mag/scale).astype(np.uint8)
	binary_output = np.zeros_like(eightbit)
	binary_output[(eightbit > thresh[0]) & (eightbit < thresh[1])] = 1
	return binary_output

def hls_select(img, sthresh=(0, 255), lthresh=()):
	hls_img = cv.cvtColor(img, cv.COLOR_RGB2HLS)
	L = hls_img[:, :, 1]
	S = hls_img[:, :, 2]
	binary_output= np.zeros_like(S)
	binary_output[(S >=sthresh[0]) & (S <=sthresh[1])
				& (L > lthresh[0]) & (L <=lthresh[1])] = 1
	return binary_output


plt.imshow(hls_select(image, sthresh=(140,255), lthresh=(120,255)), cmap='gray')
#plt.show()
plt.cla()


# Use an ensemble of filters
# threshold of sobel gradient
# + threshold of gradient magnitude and direction
def binary_pipeline(img):
	img_copy = cv.GaussianBlur(img, (3,3),0)
	
	# Color Saturation channel in HSV
	s_binary = hls_select(img_copy, sthresh=(140,255), lthresh=(120,255))
	
	# Sobel
	x_binary = abs_sobel_thresh(img_copy, thresh=(25,200))
	y_binary = abs_sobel_thresh(img_copy, thresh=(25,200), orient='y')

	xy = cv.bitwise_and(x_binary, y_binary)

	# magnitude and direction
	mag_binary = mag_thresh(img_copy, sobel_kernel=3, thresh=(30,100))
	dir_binary = mag_thresh(img_copy, sobel_kernel=3, thresh=(0.8, 1.2))

	# Stack each channel
	gradient = np.zeros_like(s_binary)
	gradient[((xy == 1) | (mag_binary == 1) & (dir_binary == 1))] = 1
	
	# Apply the image filter
	final_binary = cv.bitwise_or(s_binary, gradient)
	
	return final_binary

def warp_image(img):
	image_size = (img.shape[1], img.shape[0])
	x = img.shape[1]
	y = img.shape[0]

	source_points = np.float32([
		[0.117 * x, y],
		[(0.5 * x) - (x*0.078), (2/3)*y],
		[(0.5 * x) + (x*0.078), (2/3)*y],
		[x - (0.117 * x), y]
		])

	destination_points = np.float32([
		[0.25 * x, y],
		[0.25 * x, 0],
		[x - (0.25 * x), 0],
		[x - (0.25 * x), y]
		])
	perspective_transform = cv.getPerspectiveTransform(source_points, destination_points)
	inverse_perspective_transform = cv.getPerspectiveTransform(destination_points, source_points)

	warped_img = cv.warpPerspective(img, perspective_transform, image_size, flags=cv.INTER_LINEAR)
	return warped_img, inverse_perspective_transform
result = binary_pipeline(image)
birdseye_result, inverse_perspective_transform = warp_image(result)
plt.close('all')
plt.figure()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
f.tight_layout()
image_size = (image.shape[1], image.shape[0])
x = image.shape[1]
y = image.shape[0] 
print(image_size)
source_points = np.int32([
			[0.117 * x, y],
			[(0.5*x) - (x*0.078), (2/3)*y],
			[(0.5*x) + (x*0.078), (2/3)*y],
			[x - (0.117*x), y]
			])

poly = cv.polylines(image, [source_points], True, (255,0,0), 5)
ax1.imshow(poly, cmap='gray')
ax2.imshow(birdseye_result, cmap='gray')
plt.tight_layout()
#plt.show()
plt.cla()
plt.close('all')


# Now detect lane lines by using histogram
histogram = np.sum(birdseye_result[int(birdseye_result.shape[0]/2):, :], axis=0)
plt.figure()
plt.plot(histogram)
plt.savefig('saved/lane_histogram.png')
#plt.show()
plt.cla()
plt.close()

def track_lanes_initialize(binary_warped):
	global window_search
	histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):, :], axis=0)
	# Create an output image to draw on and visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
	#plt.imshow(out_img)
	#plt.show()	
	# We need max for each half of the histogram.
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# choose the number of sliding windows
	nwindows = 9
	window_height = np.int(binary_warped.shape[0]/nwindows)

	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	leftx_current = leftx_base
	rightx_current = rightx_base

	margin = 100
	minpix = 40
	left_lane_inds = []
	right_lane_inds = []

	for window in range(nwindows):
		# Identify window boundaries in x and y 
		win_y_low = int(binary_warped.shape[0] - (window+1)*window_height)
		win_y_high = int(binary_warped.shape[0] - (window*window_height))
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		
		cv.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 3)
		cv.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 3)
		plt.imshow(out_img)
		plt.show()	
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) 			&(nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &(nonzerox < win_xright_high)).nonzero()[0]

		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)

		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			print("current left ", leftx_current)
		if len(good_right_inds) > minpix:
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
			print("current right ", rightx_current)
	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzerox[right_lane_inds]

	# Fit a second order polynomial to each
	# Key point
	left_fit = np.polyfit(lefty, leftx, 2)
	print(righty.size)
	print(rightx.size)
	right_fit = np.polyfit(righty, rightx, 2)


	# Generate x and y for plotting
	ploty = np.linspace(0, binary_warped.shape[0] -1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1] * ploty + left_fit[2]
	right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	margin = 100
	left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy**2) + left_fit[1] * (nonzeroy) + left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
	right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy **2) + right_fit[1] * (nonzeroy) + right_fit[2] - margin)) & (nonzerox < (right_fit[0] *(nonzeroy**2) + right_fit[1]*nonzeroy+right_fit[2])))

	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	# Run the poly fit again 
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	return left_fit, right_fit

def track_lanes_initialize_demo(binary_warped):
    
    global window_search
    
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # we need max for each half of the histogram. the example above shows how
    # things could be complicated if didn't split the image in half 
    # before taking the top 2 maxes
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    # this will throw an error in the height if it doesn't evenly divide the img height
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = int(binary_warped.shape[0] - (window+1)*window_height)
        win_y_high = int(binary_warped.shape[0] - window*window_height)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 3) 
        cv.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 3)
		
        plt.imshow(out_img)
        plt.show() 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            print("left ", leftx_current)
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            print("right ", rightx_current)
            
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit,right_fit


left_fit, right_fit = track_lanes_initialize(birdseye_result)

def track_lanes_update(binary_warped, left_fit, right_fit):
	global window_search
	global frame_count
	

	
