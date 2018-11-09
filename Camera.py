import time
import cv2
import math
from Person import *
check_key = True
last_checklist = {}
pid = 1
entered = 0
exited = 0

class Camera(object):
    def __init__(self, input):
        self.input = input
        self.video = cv2.VideoCapture(self.input)
        self.w = int(self.video.get(3))
        self.h = int(self.video.get(4))
        self.rangeLeft = int(1*self.w/8)
        self.rangeRight = int(7*self.w/8)
        self.midLine = int(4*self.w/8)
        self.counter= []
        self.start_time = time.time()
    def __del__(self):
        self.video.release()

    def frameDetections(self):
        return self.video.read()

    def people_tracking(self, rects, persons):
        global pid
        global entered
        global check_key
        global last_checklist

        # for xCenter, yCenter, w, h in rects:
        for person in rects:

            xCenter, yCenter, w, h = person['rect']
            gender = person['gender']
            age = person['age']
            proj = person['project']
            new = True
            inActiveZone= xCenter in range(self.rangeLeft,self.rangeRight)

            #check every ppl location in every 5s
            if int(time.time() - self.start_time) % 10 == 0 and check_key == True:
                cur_checklist = {}
                cur_indexlist = {}
                for index, p in enumerate(persons):
                    fix_dist = math.sqrt((p.getX())**2 + (p.getY())**2)
                    cur_checklist["pid" + str(p.getId())] = fix_dist
                    cur_indexlist["pid" + str(p.getId())] = index

                check_key = False

                for key in last_checklist.keys():
                    if cur_checklist[key] == last_checklist[key]:
                        print("[INFO] pop : ", key)
                        persons.pop(cur_indexlist[key])
                        cur_checklist.pop(key)
                        print("[INFO] left pid = " , cur_checklist.keys())

                last_checklist = cur_checklist


            elif int(time.time() - self.start_time) % 5 != 0:
                check_key = True

            for index, p in enumerate(persons):

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
                        entered += 1 #count ppl in the area
                        try:
                            last_checklist.pop("pid" + str(p.getId())) #pop last frame dict id --- > 1st then pop list
                        except Exception as e:
                            pass
                        persons.pop(index)
                        for p in persons:
                            print("[INFO] left pid = ", p.getId())
            #make sure the total persons number wont excess the bounding box
            if new == True and inActiveZone and len(persons) + 1 <= len(rects) :
                print("[INFO] new person " + str(pid))
                p = Person(pid, xCenter, yCenter)
                persons.append(p)
                pid += 1
