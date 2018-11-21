import cv2
import numpy as np


class AreaOfInterest(object):
    def __init__(self,box_points):
        self.BoundingBoxes = box_points
        self.timeLapsed = np.zeros((1,len(box_points)))
        self.counter = np.zeros((1,len(box_points)))
        self.focusRadi = 10

    def check_box(self, points):
        for point in points:
            counter = 0
            for i in range(len(self.BoundingBoxes)):
                if (point[0] > self.BoundingBoxes[i,0]) and (point[1] < self.BoundingBoxes[i,1]) and (point[0] < self.BoundingBoxes[i,2]) and (point[1] > self.BoundingBoxes[i,3]):
                    counter = counter + 1
                    if counter < 2:
                        self.timeLapsed[0,i] = self.timeLapsed[0,i] + 1
                    self.counter[0,i] = self.counter[0,i] + 1

    def check_project(self,point):
        length = len(self.BoundingBoxes)
        for i in range(length):
            # # IDEA: check the area of interest from the end pointer of the user, project with largest area is where the user interested at
            if (point[0] > self.BoundingBoxes[i,0]) and (point[1] < self.BoundingBoxes[i,1]) and (point[0] < self.BoundingBoxes[i,2]) and (point[1] > self.BoundingBoxes[i,3]):
                return i

    def draw_bounding_box(self,frame):
        for i in self.BoundingBoxes:
            cv2.line(frame,(int(i[0]),int(i[1])),(int(i[0]),int(i[3])), [125,125,125], 2)
            cv2.line(frame,(int(i[0]),int(i[1])),(int(i[2]),int(i[1])), [125,125,125], 2)
            cv2.line(frame,(int(i[2]),int(i[3])),(int(i[0]),int(i[3])), [125,125,125], 2)
            cv2.line(frame,(int(i[2]),int(i[3])),(int(i[2]),int(i[1])), [125,125,125], 2)

    def update_info(self,frame):
        length = len(self.BoundingBoxes)
        for i in range(length):
            cv2.putText(frame, 'Time stayed: '+str(self.timeLapsed[0,i]), (int(self.BoundingBoxes[i,2]) - 200, int(self.BoundingBoxes[i,3]) + 25),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (200, 10, 10), 1)
            cv2.putText(frame, 'people viewing: '+str(self.counter[0,i]), (int(self.BoundingBoxes[i,2]) - 200, int(self.BoundingBoxes[i,3]) + 50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (200, 10, 10), 1)
        self.counter = np.zeros((1,len(self.BoundingBoxes)))
