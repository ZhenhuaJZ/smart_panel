import time
import cv2
import math
from Person import *

class Camera(object):
    def __init__(self, input):

        #clock
        self.trustworth_time = 10 #moing queue1 to querue2 every 10s
        self.sys_clear_time = self.trustworth_time/2
        self.no_detection_time = time.time()

        self.input = input
        self.video = cv2.VideoCapture(self.input)
        # self.video.set(3,1280)
        # self.video.set(4,720)
        self.w = int(self.video.get(3))
        self.h = int(self.video.get(4))
        self.rangeLeft = int(1*self.w/8)
        self.rangeRight = int(7*self.w/8)
        self.midLine = int(4*self.w/8)

        self.start_time = time.time()
        self.entered = 0

        #queue1
        self.persons= []
        self.pid = 1
        self.check_key = True
        self.last_checklist = {}

        #queue2
        self.stable_persons= []
        self.stable_pid = 1
        self.stable_check_key = True
        self.stable_last_checklist = {}
        self.display_pid = []
        self.bounding_box = []

        #queue3
        self.valid_persons = []

    def __del__(self):
        self.video.release()

    def frameDetections(self):
        return self.video.read()

    def fack_preson_check(self):
        #check every ppl location in every "sys_clear_time"
        '''

        '''
        if int(time.time() - self.start_time) % self.sys_clear_time == 0 and self.check_key == True:
            cur_checklist = {}
            cur_indexlist = {}
            for index, p in enumerate(self.persons):

                fix_dist = math.sqrt((p.getX())**2 + (p.getY())**2)
                cur_checklist["pid" + str(p.getId())] = fix_dist
                cur_indexlist["pid" + str(p.getId())] = index

            self.check_key = False
            for key in self.last_checklist.keys():
                try:
                    if cur_checklist[key] == self.last_checklist[key]:
                        print("[POP] pop q1 : ", key)
                        self.persons.pop(cur_indexlist[key])
                        cur_checklist.pop(key)

                        ## DEBUG:
                        #print("[INFO] stay in q1 pid = " , [k for k in cur_checklist.keys()])

                except Exception as e:
                    pass

            self.last_checklist = cur_checklist

        elif int(time.time() - self.start_time) % self.sys_clear_time != 0:
            self.check_key = True

    def stabel_fack_preson_check(self):
        #check every ppl location in every sys_clear_time
        if int(time.time() - self.start_time) % self.sys_clear_time == 0 and self.stable_check_key == True:
            cur_checklist = {}
            cur_indexlist = {}
            for index, p in enumerate(self.stable_persons):
                fix_dist = math.sqrt((p.getX())**2 + (p.getY())**2)
                cur_checklist["pid" + str(p.getId2())] = fix_dist
                cur_indexlist["pid" + str(p.getId2())] = index

            self.stable_check_key = False

            for key in self.stable_last_checklist.keys():
                try:
                    if cur_checklist[key] == self.stable_last_checklist[key]:
                        print("[POP] pop q2 : ", key)
                        self.stable_persons.pop(cur_indexlist[key])
                        cur_checklist.pop(key)
                        self.entered += 1

                        print("[MOVE] q2 pid{} -> q3".format(p.getId2()))
                        p = self.stable_persons[cur_indexlist[key]]
                        p.updateLeavetime(time.time()-self.sys_clear_time) #update person leave time
                        self.valid_persons.append(p)

                except Exception as e:
                    pass

            self.stable_last_checklist = cur_checklist

        elif int(time.time() - self.start_time) % self.sys_clear_time != 0:
            self.stable_check_key = True

    def people_tracking(self, rects):

        '''
        if no detection longer than 3min flush system
        '''
        # if len(rects) == 0 and (time.time() - self.no_detection_time) > 180:
        #     self.persons = []
        #     self.stable_persons = []
        #     self.no_detection_time = time.time()
        #     print("[WARNING] no detection")

        self.display_pid = [] #fresh display list must here !!!!!
        self.bounding_box = []

        for person in rects:

            xCenter, yCenter, w, h = person['rect']
            gender = person['gender']
            age = person['age']
            proj = person['project']
            new = True
            inActiveZone= xCenter in range(self.rangeLeft,self.rangeRight)

            """
            queue1
            """
            self.fack_preson_check()

            for index, p in enumerate(self.persons):
                #if person stay in frame over 5s
                if time.time()- p.getEnter_t() > self.trustworth_time:
                    print("[MOVE] q1 pid{} -> q2 pid{} ".format(p.getId(), str(self.stable_pid)))
                    p.updatePid2(self.stable_pid)
                    self.stable_persons.append(p)
                    self.persons.pop(index) #pop person from the persons list
                    self.stable_pid += 1

                dist = math.sqrt((xCenter - p.getX())**2 + (yCenter - p.getY())**2)
                p.updateAttris(age, gender, proj)

                if dist <= w and dist <=h:
                    if inActiveZone:
                        new = False
                        p.updateCoords(xCenter,yCenter)
                        break
                    else:
                        try:
                            self.persons.pop(index)
                            print("[POP] person {} removed from q1".format(str(p.getId())))
                            self.last_checklist.pop("pid" + str(p.getId())) #pop last frame dict id --- > 1st then pop list
                        except Exception as e:
                            pass

            """
            queue2
            """
            self.stabel_fack_preson_check()

            for index, p in enumerate(self.stable_persons):
                dist = math.sqrt((xCenter - p.getX())**2 + (yCenter - p.getY())**2)
                p.updateAttris(age, gender, proj)

                if dist <= w and dist <=h:
                    if inActiveZone:
                        new = False
                        self.bounding_box.append([xCenter, yCenter, w, h])
                        self.display_pid.append(p.getId2())
                        p.updateCoords(xCenter,yCenter)
                        break
                    else:
                        print("[POP] person {} removed from q2".format(str(p.getId2())))
                        self.entered += 1 #count ppl in the area
                        try:
                            self.stable_last_checklist.pop("pid" + str(p.getId2())) #pop last frame dict id --- > 1st then pop list
                        except Exception as e:
                            pass

                        print("[MOVE] q2 pid{} -> q3".format(p.getId2()))
                        self.stable_persons.pop(index)
                        p.updateLeavetime(time.time()) #update person leave time
                        self.valid_persons.append(p)

            #make sure the total persons number wont excess the bounding box
            if new == True and inActiveZone and len(self.persons) + len(self.stable_persons) + 1 <= len(rects) :
                print("[CREAT] new person " + str(self.pid))
                enter_t = time.time()
                p = Person(self.pid, xCenter, yCenter, enter_t)
                self.persons.append(p)
                self.pid += 1
