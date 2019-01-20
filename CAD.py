import numpy as np
from pointerNetwork import *

import random


RESOLUTION = 64

import torch
import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        def conv_block(in_channels, out_channels, p=True):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2))

        self.inputImageDimension = RESOLUTION

        # channels for hidden
        hid_dim = 64
        z_dim = 64

        self.encoder = nn.Sequential(
            conv_block(1, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            Flatten()
        )

        self.outputDimensionality = 1024

    def forward(self, v):
        if isinstance(v, list): v = np.array(v)
        if len(v.shape) == 2: squeeze = True
        v = torch.tensor(v).unsqueeze(0)
        if squeeze: v = v.unsqueeze(0)
        v = self.encoder(v.float())
        if squeeze: v = v.squeeze(0)
        return v
    
class CSG():
    lexicon = ['+','-','t','c','r'] + list(range(64))

    def execute(self): return self.render()
    
    def render(self, w=None, h=None):
        w = w or RESOLUTION
        h = h or RESOLUTION
        
        a = np.zeros((w,h))
        for x in range(w):
            for y in range(h):
                if np.array([x,y]) in self:
                    a[x,y] = 1
        return a

    @staticmethod
    def parseLine(tokens):
        if tokens[0] == '+':
            if len(tokens) != 3: return None
            if not isinstance(tokens[2],CSG): return None
            if not isinstance(tokens[1],CSG): return None
            return Union(tokens[0],tokens[1])
        if tokens[0] == '-':
            if len(tokens) != 3: return None
            if not isinstance(tokens[2],CSG): return None
            if not isinstance(tokens[1],CSG): return None
            return Difference(tokens[0],tokens[1])
        if tokens[0] == 't':
            if len(tokens) != 4: return None
            if not isinstance(tokens[3],CSG): return None
            try:
                return Translation(np.array([int(tokens[1]),int(tokens[2])]),
                                   tokens[3])
            except: return None
        if tokens[0] == 'r':
            if len(tokens) != 3: return None
            try:
                return Rectangle(int(tokens[1]),
                                 int(tokens[2]))
            except: return None
        if tokens[0] == 'c':
            if len(tokens) != 2: return None
            try: return Circle(int(tokens[1]))
            except: return None
        return None

    def toSLC(self):
        lines = []
        self._toLine(lines)
        return lines

    def calculateTrace(self):
        lines = self.toSLC()
        trace = []
        for l in lines:
            l = [trace[t.i] if isinstance(t,Pointer) else t for t in l ]
            trace.append(self.parseLine(l))
        return trace

            
        

class Rectangle(CSG):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def _toLine(self, lines):
        key = ('r',str(self.w),str(self.h))
        if key in lines: return Pointer(lines.index(key), len(lines))
        lines.append(key)
        return Pointer(len(lines) - 1)

    def __contains__(self, p):
        return p[0] >= 0 and p[1] >= 0 and \
            p[0] < self.w and p[1] < self.h

class Circle(CSG):
    def __init__(self, r):
        self.r = r

    def _toLine(self, lines):
        key = ('c',str(self.r))
        if key in lines: return Pointer(lines.index(key), len(lines))
        lines.append(key)
        return Pointer(len(lines) - 1)

    def __contains__(self, p):
        return p[0]*p[0] + p[1]*p[1] <= self.r

class Translation(CSG):
    def __init__(self, p, child):
        self.v = p
        self.child = child

    def _toLine(self, lines):
        key = ('t',str(self.v[0]),str(self.v[1]),self.child._toLine(lines))
        if key in lines: return Pointer(lines.index(key), len(lines))
        lines.append(key)
        return Pointer(len(lines) - 1)


    def __contains__(self, p):
        return (p - self.v) in self.child

class Union(CSG):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def _toLine(self, lines):
        key = ('+',self.a._toLine(lines),self.b._toLine(lines))
        if key in lines: return Pointer(lines.index(key))
        lines.append(key)
        return Pointer(len(lines) - 1)

    def __contains__(self, p):
        return p in self.a or p in self.b

class Difference(CSG):
    def __init__(self, a, b):
        self.a, self.b = a, b
    def _toLine(self, lines):
        key = ('-',self.a._toLine(lines),self.b._toLine(lines))
        if key in lines: return Pointer(lines.index(key))
        lines.append(key)
        return Pointer(len(lines) - 1)

    def __contains__(self, a, b):
        return p in self.a and (not (p in self.b))

def randomScene(resolution=64):
    def quadrilateral():
        w = random.choice(range(int(resolution/2))) + 3
        h = random.choice(range(int(resolution/2))) + 3
        x = random.choice(range(resolution - w))
        y = random.choice(range(resolution - h))
        return Translation(np.array([x,y]),
                           Rectangle(w,h))

    def circular():
        r = random.choice(range(int(resolution/4))) + 2
        x = random.choice(range(resolution - r*2)) + r
        y = random.choice(range(resolution - r*2)) + r
        return Translation(np.array([x,y]),
                           Circle(r))
    s = None
    for _ in range(random.choice([1,2])):
        o = quadrilateral() if random.choice([True,False]) else circular()
        if s is None: s = o
        else: s = Union(s,o)
    return s


if __name__ == "__main__":
    m = SLCNetwork(CNN(), CSG.lexicon)
    import matplotlib.pyplot as plot

    m = CNN()
    
    for _ in range(20):
        s = randomScene(64)
        for n,l in enumerate(s.toSLC()):
            print(f"{n} := {l}")
        for t in s.calculateTrace():
            print(t)
        plot.imshow(s.render(64,64))
        print(m(s.render(64,64)).shape)
        plot.show()
        
        
