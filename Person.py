from functools import reduce
import numpy as np

class Person:
    def __init__(self, id, rect, enter_t, proj_keys, dt):

        self.id = id

        self.rect = list(rect) #[x, y, w, h]
        '''State variables for kalman tracking'''
        self.state = np.array([self.rect[0], self.rect[1], 0, 0]) # x y x^ y^
        self.covar = np.array([[dt**4/4, 0, dt**3/2, 0],
                       [0, dt**4/4, 0, dt**3/2],
                       [dt**3/2, 0, dt**2, 0],
                       [0, dt**3/2, 0, dt**2]])
        self.K = 0

        '''Variables for tracking'''
        self.isTracked = 1
        self.lostTime = 0

        self.enter_t = enter_t
        self.exit_t = -1

        self.age = -1 #age
        self.gender = -1 #gender
        self.proj = -1
        self.pose = -1

        self.avg_age = []
        self.avg_gender = []
        self.proj_view_time = proj_keys

    def getId(self):
        return self.id
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def getEnter_t(self):
        return self.enter_t
    def updatePid(self, newPid):
        self.id = newPid
    def updateCoords(self, rect):
        self.rect = rect
    def updateAttris(self, age, gender, proj):
        self.age = age
        self.gender = gender
        self.proj = proj

        if (len(self.avg_age) and len(self.avg_gender)) <= 3:
            self.avg_age.append(age)
            self.avg_gender.append(gender)
        if (len(self.avg_age) and len(self.avg_gender)) > 2:
            avg_age = reduce(lambda x, y: x + y, self.avg_age) / len(self.avg_age)
            avg_gender = reduce(lambda x, y: x + y, self.avg_gender) / len(self.avg_gender)
            self.avg_age = []
            self.avg_gender = []
            self.avg_age.append(avg_age)
            # if avg_gender >= 0.5 :avg_gender = 1
            # else : avg_gender = 0
            self.avg_gender.append(avg_gender)

        self.gender = round(reduce(lambda x, y: x + y, self.avg_gender) / len(self.avg_gender))
        self.age = round(reduce(lambda x, y: x + y, self.avg_age) / len(self.avg_age))
        for k in proj.keys() : self.proj_view_time[k] += proj[k] #update proj

    def updateLeavetime(self, leave_time):
        self.exit_t = leave_time

    def getAttris(self):
        list = [self.id, self.age, self.gender, self.enter_t, self.exit_t, int(self.exit_t - self.enter_t)]
        proj_list = [self.proj_view_time[key] for key in self.proj_view_time.keys()]
        list.extend(proj_list)
        return list

    def predictState(self, dt):
        '''Initialise kalman filter variables'''
        ## OPTIMIZE: variable initialization
        A = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        B = np.array([(dt**2/2), (dt**2/2), dt, dt])
        C = np.array([[1,0,0,0],[0,1,0,0]])
        Ez = np.array([[0.1,0],[0,0.1]])
        Ex = np.array([[dt**4/4, 0, dt**3/2, 0],
                       [0, dt**4/4, 0, dt**3/2],
                       [dt**3/2, 0, dt**2, 0],
                       [0, dt**3/2, 0, dt**2]])
        u = 1
        '''Predict current state'''
        # print("[debug] a dot state\n ", A.dot(self.state))
        Q_estimate = A.dot(self.state) + B*u
        # print("[debug] Q_estimate\n", Q_estimate)
        '''Predict covariance'''
        self.covar = np.matmul(np.matmul(A, self.covar), np.transpose(A)) + np.array(Ex)
        # print("[debug] covar\n", self.covar)
        '''Obtain kalman gain'''
        self.K = self.covar.dot(np.transpose(C)).dot(np.linalg.inv(C.dot(self.covar).dot(np.transpose(C)) + Ez))

        self.state = Q_estimate
        self.rect[0:2] = self.state[0:2]
        self.isTracked = 0

    def updateState(self, rect):
        self.rect = np.array(rect)
        z = rect[0:2]
        C = np.array([[1,0,0,0],[0,1,0,0]])
        Q_estimate = self.state + self.K.dot(z - C.dot(self.state))
        self.state = Q_estimate
        self.rect[0:2] = self.state[0:2]
