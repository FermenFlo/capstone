import cv2
from datetime import datetime

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
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        cv2.imshow('Marino', fgmask)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# release the file pointers
video.release()
cv2.destroyAllWindows()