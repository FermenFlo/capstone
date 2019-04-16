from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import urllib.request
import math
from scipy.optimize import curve_fit



class Stream:
    def __init__(self, url = 'http://149.43.156.105/mjpg/video.mjpg', base_alterations = ['crop', 'gray', 'blur']):
        self.url = url
        self.base_alterations = base_alterations # ORDER IS ORDER OF EXECUTION
        self.stream = urllib.request.urlopen('http://149.43.156.105/mjpg/video.mjpg')
        self.stream_bytes = bytes()
        self.first_frame = None
        self.frame_number = 0
        self.old_contours = []
        self.contours = []
        self.contour_cap = 20
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
        self.old_contours = self.contours.copy()
        new_contours = cv2.findContours(self.altered_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        new_contours = imutils.grab_contours(new_contours)
        
        self.contours = [Contour(c, self) for c in new_contours]
        self.contours = sorted(self.contours, key = lambda x: -x.size)[:self.contour_cap]
        
        self._optimize_f1()
        
        self.contours = [c for c in self.contours if c.validate()]
        self.contours = [c.inherit() for c in self.contours]
        
            
        print([c.direction for c in self.contours])
        
        
    def _optimize_f1(self):
        pass
        
        
    def draw_contours(self):
        for c in self.contours:
            cv2.rectangle(self.current_frame, c.TL, c.BR, (0, 0, 255), 2)
            cv2.rectangle(self.clean_frame, c.TL, c.BR, (0, 0, 255), 2)
        
    def read_stream(self):
        while True:
            self.current_frame = self._grab_next_frame()
            self.clean_frame = self._alter_frame(self.current_frame.copy(), 'crop')
            
            for alteration in self.base_alterations:
                self.current_frame = self._alter_frame(self.current_frame, alteration)
            
            
            self.altered_frame = self.current_frame.copy()
            
            for alteration in ['diff', 'thresh']:
                self.altered_frame = self._alter_frame(self.altered_frame, alteration)


            self._update_contours()
            self.draw_contours()
            cv2.imshow("Frame", self.clean_frame)
            cv2.imshow("Altered Frame", self.altered_frame)
            
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
        self.min_size = 150
        self.max_size = np.inf
        self.precision_threshold = .5
        self.recall_threshold = 1 / (self.stream.contour_cap)
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
        self.altered_area = self.stream.altered_frame[self.y: self.y + self.h, self.x: self.x + self.w]
        self.precision = self._get_precision()
        self.recall = self._get_recall()
        self.f1 = (2 * self.precision * self.recall) / (self.precision + self.recall)
        
        # Inter-frame attributes
        self.initial_frame_number = self.stream.frame_number
        self.points = [self.center]
        self.direction = np.polyfit([x[0] for x in self.points], [x[1] for x in self.points], 1)[0] \
                             if len(set(self.points)) > 1 else 0
        print(len(self.points))
        self.merge_contour = None
    
    def _get_precision(self):
        TP = sum([1 for lst in self.altered_area for x in lst if x])
        FP = sum([1 for lst in self.altered_area for x in lst if not x])
        
        return TP / (TP + FP)
        
    def _get_recall(self):
        
        TP = sum([1 for lst in self.altered_area for x in lst if x])
        num_pos = sum([1 for lst in self.stream.altered_frame for x in lst if x])
        
        num_captured_pos = sum([1 for lst in self.altered_area for x in lst if x])
        
        FN = num_pos - num_captured_pos
        
        return TP / (TP + FN)
        
        
    def get_closest_neighbor(self, contours):
        dist = np.inf
        neightbor = None
        
        for c in contours:
            c_dist = math.hypot(self.center[0] - c.center[0], self.center[1] - c.center[1])
            
            if c_dist <= dist:
                self.neighbor_dist , self.neighbor = c_dist, c
                
                
    def inherit(self):
        '''This method handles inheriting information from past contours.
           For instance, this will handle inheriting past center points from the previous frame'''
        dist_thresh = 50
    
        old_contours = self.stream.old_contours
        parent_contour = None

        closest = (np.inf, None)
        for old_c in old_contours:
            dist = math.hypot(self.center[0] - old_c.center[0], self.center[1] - old_c.center[1])

            if dist < closest[0]:
                closest = (dist, old_c)
                
        if closest[0] < dist_thresh:
            parent_contour = closest[1]

        # Things to inherit from parent_contour:
        if parent_contour:
            new_contour = Contour(self.contour, self.stream)
            new_contour.initial_frame_number = parent_contour.initial_frame_number
            new_contour.points = parent_contour.points + self.points
            new_contour.age = parent_contour.age + self.age
            
            return new_contour
            
        else:
            return self
            
        



    # get closest neighbor via time, distance, and direction
    # if contours are close enough, turn them into one contour
    def merge_with(self, contour, mock = False):
        '''Given another contour, merge them together sensibly
           and return the newly formed contour'''
        dist_tresh = 50
        age_thresh = 0
        dir_thresh = 0
        dist_diff = math.hypot(self.center[0] - contour.center[0], self.center[1] - contour.center[1])
        age_diff = self.age - contour.age
        dir_diff = self.direction - contour.direction

        if dist_diff < dist_tresh and age_diff < age_thresh and dir_diff < dir_thresh:
            TL = (min(self.TL[0], contour.TL[0]), min(self.TL[1], contour.TL[1]))
            TR = (max(self.TL[0], contour.TL[0]), min(self.TL[1], contour.TL[1]))
            BL = (min(self.TL[0], contour.TL[0]), max(self.TL[1], contour.TL[1]))
            BR = (max(self.TL[0], contour.TL[0]), max(self.TL[1], contour.TL[1]))

            self.points = [self.center]
            self.points.append(self.center)
            #self.direction = ((curve_fit(fit_func, self.points[:][0], self.points[:][1]))[0][0])
            
    
    def validate(self):
        if self.precision < self.precision_threshold:
            return False
            
        elif self.recall < self.recall_threshold:
            return False
        
        elif self.size < self.min_size or self.size > self.max_size:
            return False
            
        elif self.age > self.stream.max_contour_age: # If this contour too old, delete it
            return False
            
        elif self.merge_contour: # If this contour is going to be merged, merge it and then delete it
            #self.merge_with(self.merge_contour)
            return False
        
        else:
            return True
            
            
    def delete(self):
        old_len = len(self.stream.contours)
        self.stream.contours.remove(self)
        #print('removed {}_{}'.format(id(self), self.size))

            
            
if __name__ == '__main__':
    
    stream = Stream()
    stream.read_stream()


