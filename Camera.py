import time
import cv2
import math
import sys
import numpy as np
from Person import *
from scipy.optimize import linear_sum_assignment

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
                cur_checklist["pid" + str(p.getId())] = fix_dist
                cur_indexlist["pid" + str(p.getId())] = index

            self.stable_check_key = False

            for key in self.stable_last_checklist.keys():
                try:
                    if cur_checklist[key] == self.stable_last_checklist[key]:

                        print("[MOVE] Q2 pid{} -> Q3".format(p.getId()))
                        p = self.stable_persons[cur_indexlist[key]]
                        p.updateLeavetime(time.time()-self.sys_clear_time) #update person leave time
                        self.stable_persons.pop(cur_indexlist[key])
                        cur_checklist.pop(key)
                        self.face_pool.pop(key) #clear face pool
                        self.entered += 1
                        self.valid_persons.append(p)

                except Exception as e:
                    pass

            self.stable_last_checklist = cur_checklist

        elif int(time.time() - self.start_time) % self.sys_clear_time != 0:
            self.stable_check_key = True

    def people_tracking(self, rects, dt):
        '''
        if no detection longer than 3min flush system
        '''
        # if len(rects) == 0 and (time.time() - self.no_detection_time) > 180:
        #     self.persons = []
        #     self.stable_persons = []
        #     self.no_detection_time = time.time()
        #     print("[WARNING] no detection")
        self.display_pid = [] #fresh display list must here !!!!!

        '''Update current status of the people base on previous state of each person'''
        for person in self.persons:
            print("[debug] bef pred\n", person.state)
        for person in self.persons:
            person.predictState(dt)
            print("[debug] aft pred\n", person.state)
        for person in self.stable_persons:
            person.predictState(dt)

        '''Track people who are just moved in'''
        '''Generate cost matrix'''
        costMat = []
        locMat = []
        rectCenters = []
        try:
            for person in self.persons:
                loc = person.state[0:2] # Extract x and y location of the person
                locMat.append(loc)
                cost = []
                for dect_person in rects:
                    center = dect_person['rect'][0:2]
                    rectCenters.append(center)
                    cost.append(math.sqrt((center[0] - loc[0])**2 + (center[1] - loc[1])**2))
                costMat.append(cost)
            costMat = np.array(costMat)
            print("[debug] costMat\n", costMat)
            print("[debug] locMat\n", locMat)
            print("[debug] recCenter\n", rectCenters)
        except Exception as e:
            print(e)
            raise

        '''Solve for minimal cost using hangarian algorithm'''
        try:
            if len(costMat) is not 0:
                row_ind, col_ind = linear_sum_assignment(costMat)
                print("[debug] row, col :{} {}".format(row_ind,col_ind))
            else:
                row_ind = []
                col_ind = []
        except Exception as e:
            print(e)
            pass
        # print("[debug] row, ind :{} {}".format(row_ind,col_ind))

        '''Given assigned row and col index, assign and update person'''
        for i, row_id in enumerate(row_ind):
            try:
                # DEBUG: id get swapped due to the cost function
                person = self.persons[row_id]
                rect = rects[col_ind[i]]
                print("[debug] person center {}".format(person.state[0:2]))
                print("[debug] person selected {}, {}".format(row_ind, rect['rect'][0:2]))

                person.updateAttris(rect['age'], rect['gender'], rect['project'])
                person.updateState(rect['rect'][0:2])
                self.display_pid.append([person.getId(), (10, 10, 200), col_ind[i]])

            except Exception as e:
                print("[error] --update val--", e)
                pass

        '''Check for new box'''
        try:
            boxes = [i for i in range(len(rects))]
            box = list(set(boxes) - set(col_ind))
            print(box)
            if len(box):
                for i in box:
                    enter_t = time.time()
                    rect = rects[i]
                    p = Person(self.pid, rect['rect'][0], rect['rect'][1], enter_t, rect['project'], dt)
                    self.persons.append(p)
                    self.pid += 1
        except Exception as e:
            print("[error] --newbox check---", e)
        print("*****************\n")
        # time.sleep(1)

        # for person in rects:
        #     xCenter, yCenter, w, h = person['rect']
        #     gender = person['gender']
        #     age = person['age']
        #     proj = person['project']
        #     new = True
        #     inActiveZone= xCenter in range(self.rangeLeft,self.rangeRight)
        #
        #     """
        #     queue1
        #     """
        #     self.fack_preson_check()
        #     # print("here")
        #     for index, p in enumerate(self.persons):
        #         #if person stay in frame over 5s
        #         if time.time()- p.getEnter_t() > self.trustworth_time:
        #             print("[MOVE] q1 pid{} -> q2 pid{} ".format(p.getId(), str(self.stable_pid)))
        #             p.updatePid(self.stable_pid) #replace the pid with new pid
        #             self.stable_persons.append(p)
        #             self.persons.pop(index) #pop person from the persons list
        #             self.stable_pid += 1
        #
        #         dist = math.sqrt((xCenter - p.getX())**2 + (yCenter - p.getY())**2)
        #         p.updateAttris(age, gender, proj)
        #
        #         if dist <= w and dist <=h:
        #             if inActiveZone:
        #                 new = False
        #                 self.display_pid.append([p.getId(), (10, 10, 200)]) #id, display color
        #                 p.updateCoords(xCenter,yCenter)
        #                 break
        #             else:
        #                 try:
        #                     self.persons.pop(index)
        #                     print("[POP]  pid{} removed from q1".format(str(p.getId())))
        #                     self.last_checklist.pop("pid" + str(p.getId())) #pop last frame dict id --- > 1st then pop list
        #                 except Exception as e:
        #                     pass
        #     """
        #     queue2
        #     """
        #     self.stabel_fack_preson_check()
        #
        #     '''for each people, predict current state base on prev state'''
        #     for person in self.persons:
        #         try:
        #             person.predictState(dt)
        #         except Exception as e:
        #             print(e)
        #     '''Match predicted state with people that are detected'''
        #
        #     '''If more than one person is not found, update only predict current state and track loss time'''
        #
        #     #make sure the total persons number wont excess the bounding box
        #     if new == True and inActiveZone and len(self.persons) + len(self.stable_persons) + 1 <= len(rects) :
        #         print("[CREAT] new pid" + str(self.pid))
        #         enter_t = time.time()
        #         p = Person(self.pid, xCenter, yCenter, enter_t, proj, dt)
        #         self.persons.append(p)
        #         self.pid += 1
