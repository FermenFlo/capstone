import cv2
from datetime import datetime
from constants import SLICE_INDICES

video = cv2.VideoCapture()
# curl -o stream2.avi http://155.33.224.29:8080/4/video.cgi

url = 'http://155.33.224.29:8080/4/video.cgi'
video.open('stream')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2()

while(video.isOpened()):
    
    ret, frame = video.read()
        
    if ret:
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        cropped_frame = frame[:SLICE_INDICES[0], SLICE_INDICES[1]:]

        
        cv2.imshow('Marino', cropped_frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# release the file pointers
video.release()
cv2.destroyAllWindows()