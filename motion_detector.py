#Author: Gabriel Oliveira

from imutils.video import VideoStream
import datetime
import imutils
import time
import cv2
import numpy as np


vs = cv2.VideoCapture(0) #Read the Webcam or VideoPath
text = ""
firstFrame = None

while True:
	frame = vs.read()
	frame = frame[1]

	if frame is None:
		break

	frame = imutils.resize(frame, width=500) #Resize the image
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Tranform in Grayscale
	eq = cv2.equalizeHist(gray) #Equalize the histogram
	gray = cv2.GaussianBlur(gray, (21, 21), 0) #Make the GaussianBlur with kernel 21x21

	if firstFrame is None:
		firstFrame = gray #Receive the FistGrame
		continue

	frameDelta = cv2.absdiff(firstFrame, gray) #Diff the instant frame with the fistFrame
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1] #Threshold the frame

	thresh = cv2.dilate(thresh, None, iterations=2) #Dilate the frame
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Find Contours
	cnts = imutils.grab_contours(cnts) #Convert length the contours tuple

	frameDelta = cv2.cvtColor(frameDelta, cv2.COLOR_GRAY2BGR) #Convert to 3 channels
	thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR) #Convert to 3 channels
	eq = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR) #Convert to 3 channels
	resultado = np.vstack([np.hstack([frame, thresh]), np.hstack([eq, frameDelta]) ]) #Create a concatenation

	for c in cnts:
		if cv2.contourArea(c) < 1000: #Set the value of Contours
			continue

		(x, y, w, h) = cv2.boundingRect(c)
		print(x, y, w, h) #Print the bouding box
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		resultado = np.vstack([np.hstack([frame, thresh]), np.hstack([eq, frameDelta]) ])
		text = "Alerta!"
		hora = datetime.datetime.now()
		hora = hora.strftime("%H.%M.%S")
		#cv2.imwrite('/Users/gabrieloliveira/Downloads/basic-motion-detection/'+hora+'-Alerta.jpg', resultado)
	
	#Put text on frame
	cv2.putText(frame, "Status da Area: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
	
	cv2.imshow("Live", resultado)
	#cv2.imshow("Live", frame)
	#cv2.imshow("Equalizada", eq)
	#cv2.imshow("Thresh", thresh)
	#cv2.imshow("Delta", frameDelta)

	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

vs.release()
cv2.destroyAllWindows()