from modules.centroidtracker import CentroidTracker
from modules.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True,
                help='path to Caffe \'deploy\' prototxt file')
ap.add_argument("-co", "--config", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
                help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
                help="# of skip frames between detections")
args = vars(ap.parse_args())


classes = ["person"]

print("[INFO] loading model...")

net = cv2.dnn.readNet(args["weights"], args["config"])

if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

writer = None
W = None
H = None

ct = CentroidTracker(maxDisappeared=40)
trackers = []
trackableObjects = {}

totalFrames = 0
totalDown = 0
totalUp = 0
fps = FPS().start()

while True:

	_, frame = vs.read()
	frame = cv2.resize(frame, None, fx=0.2, fy=0.2)
	H, W, _ = frame.shape
	#frame = cv2.resize(frame, (864, 480),interpolation=cv2.INTER_NEAREST)
	frame = frame[1] if args.get("input", False) else frame


	if args["input"] is not None and frame is None:
		break

	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	if W is None or H is None:
		(H, W) = frame.shape[:2]

	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)

	if totalFrames % args["skip_frames"] == 0:
		status = "Detecting"
		trackers = []
		height, width = frame.shape
		blob = cv2.dnn.blobFromImage(frame,1/255,(416,416),(0,0,0),swapRB =True,crop = False)
		net.setInput(blob)
		output_layers = net.getUnconnectedOutLayersNames()
		img = np.array(output_layers)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		layer_names_output = np.zeros_like(img)
		layer_names_output[:,:,0] = gray
		layer_names_output[:,:,1] = gray
		layer_names_output[:,:,2] = gray
		detections = net.forward(layer_names_output)

	status = "Waiting"
	rects = []

for i in np.arange(0, detections.shape[2]):
	confidence = detections[0, 0, i, 2]
	if confidence > args["confidence"]:
		idx = int(detections[0, 0, i, 1])
		if classes[idx] != "person":
			continue
		box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
		(startX, startY, endX, endY) = box.astype("int")
		tracker = dlib.correlation_tracker()
		rect = dlib.rectangle(startX, startY, endX, endY)
		tracker.start_track(rgb, rect)
		trackers.append(tracker)

	else:
		# loop over the trackers
		for tracker in trackers:
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			status = "Tracking"

			# update the tracker and grab the updated position
			tracker.update(rgb)
			pos = tracker.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# add the bounding box coordinates to the rectangles list
			rects.append((startX, startY, endX, endY))

	# draw a horizontal line in the center of the frame -- once an
	# object crosses this line we will determine whether they were
	# moving 'up' or 'down'
	cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# check to see if a trackable object exists for the current
		# object ID
		to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
		if to is None:
			to = TrackableObject(objectID, centroid)

		# otherwise, there is a trackable object so we can utilize it
		# to determine direction
		else:
			# the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

			# check to see if the object has been counted or not
			if not to.counted:
				# if the direction is negative (indicating the object
				# is moving up) AND the centroid is above the center
				# line, count the object
				if direction < 0 and centroid[1] < H // 2:
					totalUp += 1
					to.counted = True

				# if the direction is positive (indicating the object
				# is moving down) AND the centroid is below the
				# center line, count the object
				elif direction > 0 and centroid[1] > H // 2:
					totalDown += 1
					to.counted = True

		# store the trackable object in our dictionary
		trackableObjects[objectID] = to

		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# construct a tuple of information we will be displaying on the
	# frame
	info = [
		("Up", totalUp),
		("Down", totalDown),
		("Status", status),
	]

	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()