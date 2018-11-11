class Person:
    def __init__(self, id, x, y, enter_t):
        self.id = id
        self.id2 = -1
        self.x = x
        self.y = y

        self.enter_t = enter_t
        self.exit_t = 0

        self.a1 = -1 #gender
        self.a2 = -1 #age
        self.proj = -1

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
        self.a1 = gender
        self.a2 = age
        self.proj = proj
    def getAttris(self):
        return [self.id, self.proj, self.a1, self.a2]
