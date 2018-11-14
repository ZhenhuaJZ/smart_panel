from functools import reduce

class Person:
    def __init__(self, id, x, y, enter_t):

        self.id = id
        self.id2 = -1
        self.x = x
        self.y = y

        self.enter_t = enter_t
        self.exit_t = -1

        self.age = -1 #age
        self.gender = -1 #gender
        self.proj = -1

        self.avg_age = []
        self.avg_gender = []
        self.proj_view_time = {"a":0, "b":0, "c":0, "d":0}

    def getId(self):
        return self.id
    def getId2(self):
        return self.id2
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def getEnter_t(self):
        return self.enter_t
    def updatePid2(self, newPid):
        self.id2 = newPid
    def updateCoords(self, newX, newY):
        self.x = newX
        self.y = newY
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
        proj_a = int(self.proj_view_time['a'])
        proj_b = int(self.proj_view_time['b'])
        proj_c = int(self.proj_view_time['c'])
        proj_d = int(self.proj_view_time['d'])
        return [self.id, proj_a, proj_b, proj_c, proj_d, self.age, self.gender, self.enter_t, self.exit_t, int(self.exit_t - self.enter_t)]
        # return [self.id, self.proj, self.age, self.gender]
