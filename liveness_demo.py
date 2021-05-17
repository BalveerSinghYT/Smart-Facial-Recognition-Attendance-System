# USAGE
# python liveness_demo.py --model liveness.model --le le.pickle --detector face_detector

# import the necessary packages
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

import psycopg2
import face_recognition
from datetime import datetime

# Edited by Veer

#establishing the connection
conn = psycopg2.connect(
   database="AttendanceDB", user='postgres', password='veer', host='127.0.0.1', port= '5432'
)

#Creating a cursor object using the cursor() method
cursor = conn.cursor()

#camera_ip = 'http://172.16.2.107:8080/video'
path = 'ImageBasic'
#cap = cv2.VideoCapture(0)
images = []
image_name = []
classNames = []

mylist = os.listdir(path)       # Accessing all images in folder
print(mylist)

for cl in mylist:
    curImage = cv2.imread(f'{path}/{cl}')  # f'--' mean ImageAttendance/BillGates.jpg
    images.append(curImage)
    classNames.append(os.path.splitext(cl)[0])   # remove .jpg extension
    #print(type(os.path.splitext(cl)))
print(classNames)


def names():
    images_list = os.listdir(path)
    for cl in images_list:
        curImage = cv2.imread(f'{path}/{cl}') 
        images.append(curImage)
        image_name.append(os.path.splitext(cl)[0])
        print(image_name)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
	# Preparing SQL queries to INSERT a record into the database.
	cursor.execute('''SELECT roll_no FROM public.home_registeration where name = '%s';'''%name)
	
	#Fetching 1st row from the table
	result = cursor.fetchone()
	now = datetime.now()
	rollno = result
	# print(rollno[0])
	# print(type(rollno))
	# print(now.date())
	
	cursor.execute('''INSERT INTO home_attendance(date, status, roll_no_id)
	VALUES ('%s', 'present', %s)'''%(now.date(), rollno[0]))

	print(now.date())

	# Commit your changes in the database
	conn.commit()

	print("Records inserted........")
	
names()
encodeListKnown = findEncodings(images)
print("Encoding Complete")


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
	help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
mark = False

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 600 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=600)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	
	# Veer
	imgS = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	facesCurFrame = face_recognition.face_locations(imgS)
	encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
	

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the	
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face and extract the face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the detected bounding box does fall outside the
			# dimensions of the frame
			startX = max(0, startX)
			startY = max(0, startY)
			endX = min(w, endX)
			endY = min(h, endY)

			# extract the face ROI and then preproces it in the exact
			# same manner as our training data
			face = frame[startY:endY, startX:endX]
			face = cv2.resize(face, (32, 32))
			face = face.astype("float") / 255.0
			face = img_to_array(face)
			face = np.expand_dims(face, axis=0)

			# pass the face ROI through the trained liveness detector
			# model to determine if the face is "real" or "fake"
			preds = model.predict(face)[0]
			j = np.argmax(preds)
			label = le.classes_[j]

			# Veer
			if(label == "real"):
				for encodeFace, face_loc in zip(encodesCurFrame, facesCurFrame):
					matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
					faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
					
					matchIndex = np.argmin(faceDis)
					if matches[matchIndex]:
						
						name = classNames[matchIndex]
						if mark != True:
							markAttendance(name)
							mark = True
						else:
							continue

			# draw the label and bounding box on the frame
			label = "{}: {:.4f}".format(label, preds[j])
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)

	# show the output frame and wait for a key press
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()