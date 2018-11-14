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
    def updateLeavetime(self, leave_time):
        self.exit_t = leave_time
    def getAttris(self):
        # return [self.id, self.proj, self.a1, self.a2, self.enter_t, self.exit_t, int(self.exit_t - self.enter_t)]
        return [self.id, self.proj, self.age, self.gender]
