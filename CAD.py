import numpy as np


import random


RESOLUTION = 64

import torch
import torch.nn as nn



class CSG():
    def render(self, w=None, h=None):
        w = w or RESOLUTION
        h = h or RESOLUTION
        
        a = np.zeros((w,h))
        for x in range(w):
            for y in range(h):
                if np.array([x,y]) in self:
                    a[x,y] = 1
        return a

class Rectangle(CSG):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __contains__(self, p):
        return p[0] >= 0 and p[1] >= 0 and \
            p[0] < self.w and p[1] < self.h

class Circle(CSG):
    def __init__(self, r):
        self.r = r

    def __contains__(self, p):
        return p[0]*p[0] + p[1]*p[1] <= self.r

class Translation(CSG):
    def __init__(self, p, child):
        self.v = p
        self.child = child

    def __contains__(self, p):
        return (p - self.v) in self.child

class Union(CSG):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __contains__(self, p):
        return p in self.a or p in self.b

class Difference(CSG):
    def __init__(self, a, b):
        self.a, self.b = a, b
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
    import matplotlib.pyplot as plot
    
    for _ in range(20):
        s = randomScene(64)
        plot.imshow(s.render(64,64))
        plot.show()
        
        
