import cv2

cam = cv2.VideoCapture(1)

cv2.namedWindow("Photo")

img_counter = 0

while True:
	ret, frame = cam.read()
	cv2.imshow("Capture", frame)
	if not ret:
		break
	k = cv2.waitKey(1)

	if k%0xff == 27:
		print("Escape hit")
		break
	elif k%0xff == 32:
		img_name = "data_{}.jpg".format(img_counter)
		cv2.imwrite(img_name, frame)
		img_counter+=1
cam.release()
cv2.destroyAllWindows()
