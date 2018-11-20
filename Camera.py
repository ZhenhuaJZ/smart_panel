import time
import cv2
import math
import numpy as np
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

        self.face_pool = {}

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

        #queue3
        self.valid_persons = []

    def __del__(self):
        self.video.release()

    def frameDetections(self):
        return self.video.read()

    def fack_preson_check(self):
        #check every ppl location in every "sys_clear_time"
        if int(time.time() - self.start_time) % self.sys_clear_time == 0 and self.check_key == True:
            cur_checklist = {}
            for index, p in enumerate(self.persons):
                fix_dist = math.sqrt((p.getX())**2 + (p.getY())**2)
                cur_checklist["pid" + str(p.getId())] = [fix_dist, index]

            self.check_key = False
            for key in self.last_checklist.keys():
                try:
                    if cur_checklist[key] == self.last_checklist[key]:
                        print("[POP] pop Q1 : ", key)
                        self.persons.pop(cur_checklist[key][1])
                        self.face_pool.pop(key) #clear face pool
                        cur_checklist.pop(key)

                except Exception as e:
                    pass

            self.last_checklist = cur_checklist

        elif int(time.time() - self.start_time) % self.sys_clear_time != 0:
            self.check_key = True

    def stabel_fack_preson_check(self):
        #check every ppl location in every sys_clear_time
        if int(time.time() - self.start_time) % self.sys_clear_time == 0 and self.stable_check_key == True:
            cur_checklist = {}
            for index, p in enumerate(self.stable_persons):
                fix_dist = math.sqrt((p.getX())**2 + (p.getY())**2)
                cur_checklist["pid" + str(p.getId())] = [fix_dist, index]

            self.stable_check_key = False

            for key in self.stable_last_checklist.keys():
                try:
                    if cur_checklist[key] == self.stable_last_checklist[key]:

                        print("[P_MOVE] Q2 pid{} -> Q3".format(p.getId()))
                        p = self.stable_persons[cur_checklist[key][1]]
                        p.updateLeavetime(time.time()-self.sys_clear_time) #update person leave time
                        self.stable_persons.pop(cur_checklist[key][1])
                        cur_checklist.pop(key)
                        self.face_pool.pop(key) #clear face pool
                        self.entered += 1
                        self.valid_persons.append(p)

                except Exception as e:
                    raise

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

        for person in rects:

            xCenter, yCenter, _, _ = person['rect']
            gender = person['gender']
            age = person['age']
            proj = person['project']
            pid = person['pid'][0]
            face = person['face']

            new = True
            inActiveZone= xCenter in range(self.rangeLeft,self.rangeRight)

            """
            queue1
            """
            self.fack_preson_check()

            for index, p in enumerate(self.persons):
                #if person stay in frame over 5s
                if time.time()- p.getEnter_t() > self.trustworth_time:
                    print("[MOVE] Q1 pid{} -> Q2 pid{} ".format(p.getId(), p.getId()))
                    self.stable_persons.append(p)
                    self.persons.pop(index) #pop person from the persons list

                p.updateAttris(age, gender, proj)
                _pid = "pid" + str(p.getId())
                if _pid == pid:
                    if inActiveZone:
                        new = False
                        p.updateCoords(xCenter,yCenter)
                        self.face_pool[pid] = face #update face
                        break
                    else:
                        try:
                            print("[POP]  pid{} removed from Q1".format(str(p.getId())))
                            self.persons.pop(index)
                            self.face_pool.pop(pid) #clear face pool
                            self.last_checklist.pop(pid) #pop last frame dict id --- > 1st then pop list
                        except Exception as e:
                            pass

            """
            queue2
            """
            self.stabel_fack_preson_check()

            for index, p in enumerate(self.stable_persons):
                p.updateAttris(age, gender, proj)
                _pid = "pid" + str(p.getId())
                if _pid == pid:
                    if inActiveZone:
                        new = False
                        p.updateCoords(xCenter,yCenter)
                        self.face_pool[pid] = face #update face
                        break
                    else:
                        self.entered += 1 #count ppl in the area
                        try:
                            self.stable_last_checklist.pop(pid) #pop last frame dict id --- > 1st then pop list
                        except Exception as e:
                            pass
                        print("[MOVE] Q2 pid{} -> Q3".format(p.getId()))
                        self.stable_persons.pop(index)
                        self.face_pool.pop(pid) #clear face pool
                        p.updateLeavetime(time.time()) #update person leave time
                        self.valid_persons.append(p)

            #make sure the total persons number wont excess the bounding box
            face_pool_check = pid not in [k for k in self.face_pool.keys()] and pid != "Confuse"
            total_p_inframe = len(self.persons) + len(self.stable_persons) + 1 <= len(rects)

            if new == True and inActiveZone and face_pool_check and total_p_inframe:
                print("[CREAT] new pid" + str(self.pid))
                enter_t = time.time()
                p = Person(self.pid, xCenter, yCenter, enter_t)
                self.persons.append(p)
                self.face_pool["pid"+str(self.pid)] = face
                print("[INFO] face pool -> ", [key for key in self.face_pool.keys()])
                self.pid += 1
