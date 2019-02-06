import cv2
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

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(cropped_frame, winStride = (8, 8), padding = (40, 40), scale = 1.02)

        # draw the original bounding boxes
        if rects:
            print(rects)
  
        for (x, y, w, h) in rects:
            cv2.rectangle(cropped_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        cv2.imshow('Marino', cropped_frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# release the file pointers
video.release()
cv2.destroyAllWindows()