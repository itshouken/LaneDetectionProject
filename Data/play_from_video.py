import numpy as np
import cv2

cap = cv2.VideoCapture('output.avi')

# Define the codec and create VideoWriter object

while cap.isOpened():
	ret, frame = cap.read()
	#print(frame.shape)
	if ret == True:
		#frame =cv2.flip(frame, 0 )

	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break


cap.release()
cv2.destroyAllWindows()
