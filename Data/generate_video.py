import numpy as np
import cv2
import datetime

cap = cv2.VideoCapture(1)
currentDT = datetime.datetime.now()
#print(str(currentDT))
# Define the codec and create VideoWriter object

video_filename = 'DrivingData' + currentDT.strftime("%H%M_%m%d") + ".avi"
print(video_filename)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

while cap.isOpened():
	ret, frame = cap.read()
	#print(frame.shape)
	if ret == True:
		frame =cv2.flip(frame, 0 )
		frame =cv2.flip(frame, 1 )
		out.write(frame)

	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break

out.release()

cap.release()
cv2.destroyAllWindows()
