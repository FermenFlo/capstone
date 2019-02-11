# HOP ON CONNNORS BRANCH AND RESTUCTURE CODE TO CLASS STRUCTURE, THEN WORK ON GEETTING BOUNDING BOX DIRECTION OF MOVEMENT OR ESTIMATE NUMBER OF PEOPLE IN BOUNDING BOX

import cv2
import numpy as np
import imutils
from imutils import paths
from imutils.object_detection import non_max_suppression
from datetime import datetime
from constants import SLICE_INDICES

video = cv2.VideoCapture()
# curl -o stream2.avi http://155.33.224.29:8080/4/video.cgi

url = 'http://155.33.224.29:8080/4/video.cgi'
video.open('stream')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while(video.isOpened()):
    
    ret, frame = video.read()
    
        
    if ret:
#         fgmask = fgbg.apply(frame)
#         fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        cropped_frame = frame[:SLICE_INDICES[0], SLICE_INDICES[1]:]

    
        # load the image and resize it to (1) reduce detection time
        # and (2) improve detection accuracy
        image = cropped_frame
        image = imutils.resize(image, width = min(400, image.shape[1]))
        orig = image.copy()


        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
            padding=(32, 32), scale=1.05)

        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # show some information on the number of bounding boxes
        filename = 'Marino'
        print("[INFO] {}: {} original boxes, {} after suppression".format(
        filename, len(rects), len(pick)))

        # show the output images
        cv2.imshow("Before NMS", orig)
        cv2.imshow("After NMS", image)
        #cv2.imshow('Marino', cropped_frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# release the file pointers
video.release()
cv2.destroyAllWindows()