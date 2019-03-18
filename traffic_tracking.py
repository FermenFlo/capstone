# python object_movement.py
# USAGE
# python object_movement.py --video object_tracking_example.mp4

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import urllib.request

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
args = vars(ap.parse_args())

delta_list = []
h_list = []
w_list = []

min_contour_width = 50
max_contour_width = 200
min_contour_height = 120
max_contour_height = 300

# initialize the list of tracked points, the frame counter
# and the coordinate deltas
pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) = (0, 0)
direction = ""

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
	firstFrame = None
	stream = urllib.request.urlopen('http://149.43.156.105/mjpg/video.mjpg')
	bytes = bytes()
# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
while True:
	bytes += stream.read(16)
	a = bytes.find(b'\xff\xd8')
	b = bytes.find(b'\xff\xd9')
	if a != -1 and b != -1:
		jpg = bytes[a:b + 2]
		bytes = bytes[b + 2:]
		frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
		frame = frame[250:500, 250:550]
		frame = imutils.resize(frame, width=600)
		blurred = cv2.GaussianBlur(frame, (11, 11), 0)
		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

		#frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 3, 21)
		# resize the frame, convert it to grayscale, and blur it
		frame = imutils.resize(frame, width=500)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21, 21), 0)

		if firstFrame is None:
			firstFrame = gray
			continue

		center = None

		# compute the absolute difference between the current frame and
		# first frame
		frameDelta = cv2.absdiff(firstFrame, gray)
		thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

		# dilate the thresholded image to fill in holes, then find contours
		# on thresholded image
		thresh = cv2.dilate(thresh, None, iterations=2)
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
								cv2.CHAIN_APPROX_NONE)
		cnts = imutils.grab_contours(cnts)


		# only proceed if at least one contour was found
		if len(cnts) > 0:
			# find the largest contour in the mask, then useq
			# it to compute the minimum enclosing rectangle and centroid
			cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
			cnts = cnts[0:10]
			for c in cnts:
				(x, y, w, h) = cv2.boundingRect(c)
				#contour_valid = (w >= min_contour_width and w <= max_contour_width) and (h >= min_contour_height and h<= max_contour_height)
				#contour_valid = (h > w) and (w < max_contour_width)
				contour_valid = ((cv2.contourArea(c)>500))

				if not contour_valid:
					continue

				M = cv2.moments(c)
				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

				# only proceed if the radius meets a minimum size
				# draw the rectangle and centroid on the frame,
				# then update the list of tracked points
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
				pts.appendleft(center)

		# loop over the set of tracked points
		for i in np.arange(1, len(pts)):
			# if either of the tracked points are None, ignore them
			if pts[i - 1] is None or pts[i] is None:
				continue

			# check to see if enough points have been accumulated in
			# the buffer
			if counter >= 10 and i == 1 and pts[-1] is not None:
				# compute the difference between the x and y
				# coordinates and re-initialize the direction
				# text variables
				if center:
					dX = pts[-1][0] - pts[i][0]
					dY = pts[-1][1] - pts[i][1]
				else:
					dX = 0
					dY = 0
				(dirX, dirY) = ("", "")

				# ensure there is significant movement in the
				# x-direction
				if np.abs(dX) > 20:
					dirX = "East" if np.sign(dX) == 1 else "West"

				# ensure there is significant movement in the
				# y-direction
				if np.abs(dY) > 20:
					dirY = "North" if np.sign(dY) == 1 else "South"

				# handle when both directions are non-empty
				if dirX != "" and dirY != "":
					direction = "{}-{}".format(dirY, dirX)

				# otherwise, only one direction is non-empty
				else:
					direction = dirX if dirX != "" else dirY

			# otherwise, compute the thickness of the line and
			# draw the connecting lines
			thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
			cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

		# show the movement deltas and the direction of movement on
		# the frame
		#print("dx: {}, dy: {}".format(dX, dY))
		print(dX,dY)
		delta_list.append((dX,dY))
		cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
			0.65, (0, 0, 255), 3)
		cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
			(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
			0.35, (0, 0, 255), 1)

		# show the frame to our screen and increment the frame counter
		cv2.imshow("Frame", frame)
		#cv2.imshow("Thresh", thresh)
		key = cv2.waitKey(1) & 0xFF
		counter += 1

		# if the 'q' key is pressed, stop the loop
		if key == ord("q"):
			break



#print(delta_list)
#print(len(delta_list))
print(h_list)
print(w_list)


# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
#else:
#	vs.release()

# close all windows
cv2.destroyAllWindows()