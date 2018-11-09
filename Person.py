class Person:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.a1 = -1 #gender
        self.a2 = -1 #age
        self.proj = -1

    def getId(self):
        return self.id
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def updateCoords(self, newX, newY):
        self.x = newX
        self.y = newY
    def updateAttris(self, age, gender, proj):
        self.a1 = gender
        self.a2 = age
        self.proj = proj
    def getAttris(self):
        return [self.id, self.proj, self.a1, self.a2]
