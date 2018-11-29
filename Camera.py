import time
import cv2
import math
import sys
import numpy as np
from Person import *
import copy
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


    def extract_corner(self, rect):
        xCenter = rect[0]
        yCenter = rect[1]
        width = rect[2]
        height = rect[3]
        xmin = xCenter - width/2
        ymin = yCenter - height/2
        xmin_2 = xCenter + width/2
        ymin_2 = yCenter + height/2
        xmax = xmin + width
        ymax = ymin + height
        xmax_2 = xmin_2 + width
        ymax_2 = ymin_2 + height
        return [[xCenter, yCenter], [xmin, ymin], [xmax, ymax], [xmin_2, ymin_2], [xmax_2, ymax_2]], height*width/100


    def get_corner_dist(self, p_rect, d_rect, a1, a2):
        p_rect = np.array(p_rect)
        d_rect = np.array(d_rect)
        # print("[debug] stage 1\n", p_rect - d_rect)
        # print("[debug] stage 2\n", np.square(p_rect - d_rect))
        # print("[debug] stage 3\n", np.sum(np.square(p_rect - d_rect), axis = 1))
        # print("[debug] stage 4\n", np.sqrt(np.sum(np.square(p_rect - d_rect), axis = 1)))
        # print("[debug] stage 4\n", np.mean(np.sqrt(np.sum(np.square(p_rect - d_rect), axis = 1))))
        avg_dist = (np.mean(np.sqrt(np.sum(np.square(p_rect - d_rect), axis = 1))) + abs(a1-a2))/2
        # print(a1-a2)
        return avg_dist


    def check_lost_time(self, persons, valid = " "):
        for person in persons:
            x,_,_,_ = person.rect
            inActiveZone = x in range(self.rangeLeft, self.rangeRight)
            if not person.isTracked and time.time() - person.lostTime > self.sys_clear_time or not inActiveZone:
                persons.remove(person)
                print("[Pop] pop id_{}".format(person.id))
                # self.pid += 1
                # self.stable_pid += 1
                # BUG: when pop from persons, stable not update
                # BUG: when pop from stable , stable updated
                if valid is not " ":
                    print("[Move] Q2 id_{} move to Q3".format(person.id))
                    valid.append(person)
                    self.entered += 1
                    person.updateLeavetime(time.time())
                # else:
                #     self.stable_pid += 1


    def check_stayed_time(self):
        for index, person in enumerate(self.persons):
            '''if person stay in frame over 5s and is tracked'''
            if time.time()- person.getEnter_t() > self.trustworth_time and person.isTracked:
                print("[MOVE] q1 pid{} -> q2 pid{} ".format(person.getId(), str(self.stable_pid)))
                person.updatePid(self.stable_pid) #replace the pid with new pid
                self.stable_persons.append(person)
                self.persons.pop(index) #pop person from the persons list
                self.stable_pid += 1


    def kalman_tracker(self, persons, stable_type_index, rects, dt):

        '''return rectangles if there are no people in the container'''
        if not len(persons):
            return rects, []

        '''Generate cost matrix'''
        costMat = []
        locMat = []
        rectCenters = []
        try:
            for person in persons:
                loc = person.state[0:2] # Extract x and y location of the person
                locMat.append(loc)
                cost = []
                for dect_person in rects:
                    # NOTE: Use distance between four corners of the two bounding box as cost function
                    person_rect, a1 = self.extract_corner(person.rect)
                    # print("[debug] person rect:\n", person.rect)
                    dect_rect, a2 = self.extract_corner(dect_person['rect'])
                    # print("[debug] dect_p rect: \n", dect_person['rect'])
                    # print("[debug] a1 & a2: {} {}".format(a1, a2))
                    avg_dist = self.get_corner_dist(person_rect, dect_rect, a1, a2)
                    center = dect_person['rect'][0:2] # Clean
                    # rectCenters.append(center)
                    cost.append(avg_dist)
                costMat.append(cost)
            # print("[debug] costMat\n", costMat)
            # print("[debug] locMat\n", locMat)
            # print("[debug] recCenter\n", rectCenters)
        except Exception as e:
            print("[error] --cost mat--",e)
            raise

        '''Solve for minimal cost using hangarian algorithm'''
        try:
            if len(costMat) is not 0:
                row_ind, col_ind = linear_sum_assignment(costMat)
                # print("[debug] row, col :{} {}".format(row_ind,col_ind))
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
                person = persons[row_id]
                rect = rects[col_ind[i]]
                # print("[debug] person center {}".format(person.state[0:2]))
                # print("[debug] person selected {}, {}".format(row_ind, rect['rect'][0:2]))
                person.updateAttris(rect['age'], rect['gender'], rect['project'])
                person.updateState(rect['rect'])
                person.pose = rect['pose']
                person.lostTime = time.time()
                person.isTracked = 1
                if stable_type_index[row_id] is 0:
                    color = (10,10,200)
                else:
                    color = (10,200,10)
                self.display_pid.append([person.getId(), color, col_ind[i]])
            except Exception as e:
                print("[error] --update val--", e)
                pass

        return rects, list(col_ind)


    def people_tracking(self, rects, dt):
        '''Make a deep copy of the rectangles so it doesnt affect draw_detection_info'''
        rects = copy.deepcopy(rects)
        self.display_pid = []

        '''Update current status of the people base on previous state of each person'''
        try:
            for person in self.persons:
                person.predictState(dt)
                # print("[debug] aft pred\n", person.state)
            for person in self.stable_persons:
                person.predictState(dt)
        except Exception as e:
            print(e)

        '''check person stayed long enough to be captured as viewer'''
        # BUG: If there are one in red and one in green, and only one face is detected, current algorithm will have both red and green refer to the same rectangle
        ## BUG: but if delete, then what the rect suppose to be green will be red first
        try:
            track_people = []
            track_index = []
            '''combining both persons and stable persons list into one'''
            '''track index is use to identify whether a person in track_people list is stable or not'''
            for person in self.persons:
                x = 1
                track_people.append(person)
                track_index.append(0)
            for person in self.stable_persons:
                track_people.append(person)
                track_index.append(1)
            '''Track persons'''
            '''Returns original rectangle and the index which people are tracked'''
            rects, rect_index = self.kalman_tracker(track_people, track_index, rects, dt)
        except Exception as e:
            print("[error] --kalman_tracker--",e)

        '''Check for the duration of lost tracked and '''
        self.check_stayed_time()

        '''Remove all excessive persons that are not tracked for certain time'''
        self.check_lost_time(self.persons)
        self.check_lost_time(self.stable_persons, self.valid_persons)

        '''Remove tracked rectangles'''
        try:
            # print("[debug] rect len {}".format(len(rects)))
            # print("[debug] rect index bes {}".format(type(rect_index)))
            # rect_index = np.sort(rect_index)
            # print("[debug] rect index aft {}".format(rect_index))
            val = [rects[i] for i in rect_index]
            # print("[debug] val\n", val)
            for i in val:
                rects.remove(i)
        except Exception as e:
            print("[error] --remove--", e)

        '''Check for new and untracked rect'''
        try:
            for rect in rects:
                x,_,_,_ = rect['rect']
                inActiveZone = x in range(self.rangeLeft, self.rangeRight)
                if inActiveZone:
                    enter_t = time.time()
                    person = Person(self.pid, rect['rect'], enter_t, rect['project'], dt)
                    print("[CREATE] new pid_" + str(self.pid))
                    self.persons.append(person)
                    self.pid += 1
                    # self.stable_pid += 1
        except Exception as e:
            print("[error] --newbox check---", e)
        # print("*****************\n")
