import time
import cv2
import math
from Person import *

class Camera(object):
    def __init__(self, input):
        self.input = input
        self.video = cv2.VideoCapture(self.input)
        self.w = int(self.video.get(3))
        self.h = int(self.video.get(4))
        self.rangeLeft = int(1*self.w/8)
        self.rangeRight = int(7*self.w/8)
        self.midLine = int(4*self.w/8)
        self.persons= []
        self.stable_persons= []
        self.start_time = time.time()
        self.check_key = True
        self.entered = 0
        self.pid = 1
        self.stable_pid = 1
        self.last_checklist = {}

    def __del__(self):
        self.video.release()

    def frameDetections(self):
        return self.video.read()

    def people_tracking(self, rects):
        # for xCenter, yCenter, w, h in rects:
        for person in rects:

            xCenter, yCenter, w, h = person['rect']
            gender = person['gender']
            age = person['age']
            proj = person['project']
            new = True
            inActiveZone= xCenter in range(self.rangeLeft,self.rangeRight)

            #check every ppl location in every 5s
            if int(time.time() - self.start_time) % 5 == 0 and self.check_key == True:
                cur_checklist = {}
                cur_indexlist = {}
                for index, p in enumerate(self.persons):
                    fix_dist = math.sqrt((p.getX())**2 + (p.getY())**2)
                    cur_checklist["pid" + str(p.getId())] = fix_dist
                    cur_indexlist["pid" + str(p.getId())] = index

                self.check_key = False

                for key in self.last_checklist.keys():
                    if cur_checklist[key] == self.last_checklist[key]:
                        print("[INFO] pop : ", key)
                        self.persons.pop(cur_indexlist[key])
                        cur_checklist.pop(key)
                        print("[INFO] left pid = " , cur_checklist.keys())

                self.last_checklist = cur_checklist


            elif int(time.time() - self.start_time) % 5 != 0:
                self.check_key = True
    
            for index, p in enumerate(self.persons):

                #if person stay in frame over 5s
                # if time.time()- p.enter_t > 5:
                #     self.stable_pid += 1
                #     p.updatePid(self.stable_pid)
                #     self.stable_persons.append(p)

                dist = math.sqrt((xCenter - p.getX())**2 + (yCenter - p.getY())**2)
                p.updateAttris(gender, age, proj)

                if dist <= w and dist <=h:
                    if inActiveZone:
                        new = False
                        if p.getX() < self.midLine and  xCenter >= self.midLine:
                            print("[INFO] person {} going left ".format(str(p.getId())))
                            # entered += 1 #count ppl in and out
                        if p.getX() > self.midLine and  xCenter <= self.midLine:
                            print("[INFO] person {} going right ".format(str(p.getId())))
                            # exited += 1
                        p.updateCoords(xCenter,yCenter)
                        break
                    else:
                        print("[INFO] person {} removed ".format(str(p.getId())))
                        self.entered += 1 #count ppl in the area
                        try:
                            self.last_checklist.pop("pid" + str(p.getId())) #pop last frame dict id --- > 1st then pop list
                        except Exception as e:
                            pass
                        self.persons.pop(index)
                        for p in self.persons:
                            print("[INFO] left pid = ", p.getId())
            #make sure the total persons number wont excess the bounding box
            if new == True and inActiveZone and len(self.persons) + 1 <= len(rects) :
                print("[INFO] new person " + str(self.pid))
                enter_t = time.time()
                p = Person(self.pid, xCenter, yCenter, enter_t)
                self.persons.append(p)
                self.pid += 1
