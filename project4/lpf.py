import line

class LineSmoother:
    def __init__(self, alfa):
        self.data = None
        self.alfa = alfa

    def ApplyLPF(self, line):

        if self.data == None:
            self.data = line.fit
            return line.fit

        self.data = self.data + self.alfa * (line.fit - self.data)
        return self.data

class Smoother:
    def __init__(self, alfa):
        self.l = LineSmoother(alfa)
        self.r = LineSmoother(alfa)

    def ApplyLPF(self, lane):
        lane.l.fit = self.l.ApplyLPF(lane.l)
        lane.r.fit = self.r.ApplyLPF(lane.r)
        return lane
