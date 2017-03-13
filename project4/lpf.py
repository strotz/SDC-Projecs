import line
import numpy as np

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

class HeatmapSmoother:
    def __init__(self, alfa):
        self.data = None
        self.alfa = alfa

    def ApplyLPF(self, heat):
        if self.data == None:
            self.data = np.copy(heat)
            return self.data

        self.data = self.data + self.alfa * (heat - self.data)
        return np.copy(self.data)

class HeatmapAverege:
    def __init__(self, total = 5):
        self.data = []
        self.total = total

    def Apply(self, heat):
        if len(self.data) >= self.total:
            self.data.pop(0)

        res = np.copy(heat)
        for z in self.data:
            res = res + z

        self.data.append(heat)
        return res
