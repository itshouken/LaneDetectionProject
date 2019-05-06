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

