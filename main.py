from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import urllib.request
import math 


class Stream:
    def __init__(self, url = 'http://149.43.156.105/mjpg/video.mjpg', alterations = []):
        self.url = url
        self.alterations = ['crop'] + alterations # ORDER IS ORDER OF EXECUTION
        self.stream = urllib.request.urlopen('http://149.43.156.105/mjpg/video.mjpg')
        self.stream_bytes = bytes()
        self.first_frame = None
        self.contours = []
        
    def _grab_next_frame(self):
        while True:
            self.stream_bytes += self.stream.read(16)
            a = self.stream_bytes.find(b'\xff\xd8')
            b = self.stream_bytes.find(b'\xff\xd9')

            if a != -1 and b != -1:
                break

        jpg = self.stream_bytes[a:b + 2]
        self.stream_bytes = self.stream_bytes[b + 2:]
        frame = cv2.imdecode(np.fromstring(jpg, dtype = np.uint8), cv2.IMREAD_COLOR)

        if self.first_frame is None:
            self.first_frame = frame
            
            for alteration in self.alterations:
                if alteration != 'diff':
                    self.first_frame = self._alter_frame(self.first_frame, alteration)
                    
        return frame
        
        
    def _alter_frame(self, frame, alteration, kwargs = {}):
        '''Given a frame and an alteration (e.g. "gray"), apply the alteation 
           to the frame and return the new frame.
           Also accepts kwargs for the alteration fucntion if needed
           
           implemented alterations:

           - "crop": Crop the frame
           - "gray": Gray the frame
           - "blur": Blur the frame
           - "dilate": Dilate the frame
           - "erode": Erode the frame
           - "diff": Take the difference between the frame and self.first_frame
           - "thresh": Threshold the image above a certain value
           - "hvs": Not entirely sure tbh...
           '''
        
        if alteration == 'crop':
            return imutils.resize(frame[250:500, 250:550], width = 600)
        
        elif alteration == 'gray':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        elif alteration == 'blur':
            return cv2.GaussianBlur(frame, (11, 11), 0)
        
        elif alteration == 'dilate':
            dilation_kernel = np.ones((5,1)) # use a large dilation kernel to capture sparse words
            return cv2.dilate(frame, dilation_kernel, iterations = 1) # dilate
        
        elif alteration == 'erode':
            erosion_kernel = np.ones((5,5)) # use a large dilation kernel to capture sparse words
            return cv2.erode(frame, erosion_kernel, iterations = 1) # dilate
        
        elif alteration == 'diff':
            if self.first_frame is None:
                return frame
            else:
                return cv2.absdiff(frame, self.first_frame)
        
        elif alteration == 'thresh':
            return cv2.threshold(frame, 25, 255, cv2.THRESH_BINARY)[1]
        
        elif alteration == 'hsv':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        
    def _update_contours(self, frame, frame_number):
        new_contours = cv2.findContours(self.current_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        new_contours = imutils.grab_contours(new_contours)
        print(self.first_frame)
        
#         for c in new_contours:
#             if self.is_new_contour(c):
#                 continue
                
#             else:
#                 pass
        pass
    
    def read_stream(self):
        frame_number = 0
        
        while True:
            frame = self._grab_next_frame()
            

            for alteration in self.alterations:
                print(alteration)
                frame = self._alter_frame(frame, alteration)


            self.current_frame = frame
            self._update_contours(frame, frame_number)
            

            cv2.imshow("Frame", self._alter_frame(self.current_frame, 'diff'))
            
            # if the 'q' key is pressed, stop the loop
            key = cv2.waitKey(1) & 0xFF
            frame_number += 1
            if key == ord("q"):
                break
                
                
class Contour:
    
    def __init__(self, c, frame_number):
        self.contour = c
        self.frame_number = frame_number
        self.age = 0
        self.x, self.y, self.w, self.h = cv2.boundingRect(c)
        self.moment = cv2.moments(c)
        self.center = (int(self.moment["m10"] / self.moment["m00"]),
                       int(self.moment["m01"] / self.moment["m00"]))
        self.TL = (self.x, self.y)
        self.TR = (self.x, self.y + self.w)
        self.BL = (self.x + self.h, self.y)
        self.BR = (self.x + self.h, self.y + self.w)
        self.size = cv2.contourArea(c)
        self.points = [center]
        
        
    def get_closest_neighbor(self, contours):
        dist = np.inf
        neightbor = None
        for c in contours:
            c_dist = math.hypot(self.center[0] - c.center[0], self.center[1] - c.center[1])
            
            if c_dist <= dist:
                dist, neighbor = c_dist, c
                
        self.neighbor_dist = dist
        self.neighbor = c
            
    
    def merge_contours(self, contour):
        '''Given another contour, merge them together sensibly
           and return the newly formed contour'''
        pass
        

            
if __name__ == '__main__':
    
    stream = Stream(alterations = ['gray', 'diff'])
    stream.read_stream()