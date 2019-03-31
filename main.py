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
    def __init__(self, url = 'http://149.43.156.105/mjpg/video.mjpg', base_alterations = ['crop', 'gray', 'blur']):
        self.url = url
        self.base_alterations = base_alterations # ORDER IS ORDER OF EXECUTION
        self.stream = urllib.request.urlopen('http://149.43.156.105/mjpg/video.mjpg')
        self.stream_bytes = bytes()
        self.first_frame = None
        self.frame_number = 0
        self.contours = []
        self.contour_cap = 50
        self.max_contour_age = 200
        
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
            
            for alteration in self.base_alterations:
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
            dilation_kernel = np.ones((5,5)) # use a large dilation kernel
            return cv2.dilate(frame, dilation_kernel, iterations = 1) # dilate
        
        elif alteration == 'erode':
            erosion_kernel = np.ones((5,5)) # use a large dilation kernel
            return cv2.erode(frame, erosion_kernel, iterations = 1) # dilate
        
        elif alteration == 'diff':
            try: # Doesn't work if on first frame
                return cv2.absdiff(frame, self.first_frame)
            
            except:
                return frame
        
        elif alteration == 'thresh':
            return cv2.threshold(frame, 25, 255, cv2.THRESH_BINARY)[1]
        
        elif alteration == 'hsv':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        
    def _update_contours(self):
        new_contours = cv2.findContours(self.altered_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        new_contours = imutils.grab_contours(new_contours)
        
        self.contours += [Contour(c, self.frame_number) for c in new_contours]
        self.contours = self.contours[:self.contour_cap]
        
        
    def draw_contours(self):
        for c in self.contours:
            cv2.rectangle(self.current_frame, c.TL, c.BR, (0, 0, 255), 2)
        
    def read_stream(self):
        while True:
            self.current_frame = self._grab_next_frame()
            
            for alteration in self.base_alterations:
                self.current_frame = self._alter_frame(self.current_frame, alteration)
            
            
            self.altered_frame = self.current_frame.copy()
            
            for alteration in ['diff', 'thresh']:
                self.altered_frame = self._alter_frame(self.altered_frame, alteration)


            self._update_contours()
            self.draw_contours()
            cv2.imshow("Frame", self.current_frame)
            cv2.imshow("First Frame", self.first_frame)
            
            # if the 'q' key is pressed, stop the loopa
            key = cv2.waitKey(1) & 0xFF
            self.frame_number += 1
            if key == ord("q"):
                break
                
                


class Contour:
    
    def __init__(self, c, stream):
        
        # Meta attributes
        self.contour = c
        self.stream = stream
        
        # Intra-frame attributes
        self.age = 0
        self.x, self.y, self.w, self.h = cv2.boundingRect(c)
        self.moment = cv2.moments(c)
        self.center = (int(self.moment["m10"] / (self.moment["m00"] or 1)),
                       int(self.moment["m01"] / (self.moment["m00"] or 1)))
        self.TL = (self.x, self.y)
        self.TR = (self.x, self.y + self.w)
        self.BL = (self.x + self.h, self.y)
        self.BR = (self.x + self.h, self.y + self.w)
        self.size = cv2.contourArea(c)
        
        # Inter-frame attributes
        self.initial_frame_number = stream.frame_number
        self.points = [self.center]
        self.merge_contour = None
        
    def _get_age(self):
        self._age = self.stream.frame_number - self.initial_frame_number
        return self._age
    
    def set_age(self, value):
        self._age = age
        
    age = property(get_age, set_age) # override self.age with a property
        
    def get_closest_neighbor(self, contours):
        dist = np.inf
        neightbor = None
        
        for c in contours:
            c_dist = math.hypot(self.center[0] - c.center[0], self.center[1] - c.center[1])
            
            if c_dist <= dist:
                self.neighbor_dist , self.neighbor = c_dist, c


    # get closest neighbor via time, distance, and direction
    # if contours are close enough, turn them into one contour
    def merge_contours(self, contour):
        '''Given another contour, merge them together sensibly
           and return the newly formed contour'''
        c_dist = math.hypot(self.center[0] - contour.center[0], self.center[1] - contour.center[1])
        too_close_dist = 0
        if c_dist < too_close_dist:
            TL = (min(self.TL[0], contour.TL[0]), min(self.TL[1], contour.TL[1]))
            TR = (max(self.TL[0], contour.TL[0]), min(self.TL[1], contour.TL[1]))
            BL = (min(self.TL[0], contour.TL[0]), max(self.TL[1], contour.TL[1]))
            BR = (max(self.TL[0], contour.TL[0]), max(self.TL[1], contour.TL[1]))
            
            
    def delete(self):
        if self.age > self.stream.max_contour_age: # If this contour too old, delete it
            print('Contour_{}: deleting self for being too old'.format(id(self)))
            self.stream.contours.remove(self)
            
        if self.merge_contour: # If this contour is going to be merged, merge it and then delete it
            print('Contour_{}: deleting self for merging'.format(id(self)))
            self.merge_with(self.merge_contour)
            self.stream.contours.remove(self)
            
if __name__ == '__main__':
    
    stream = Stream()
    stream.read_stream()


