from program import *

from arithmeticPrimitives import *
from logicalPrimitives import *

from functools import reduce

#def _concatenate(x): return lambda y: x + y

def _left(x): return [tuple([b[0] - 1] + list(b[1:])) for b in x]
def _right(x): return [tuple([b[0] + 1] + list(b[1:])) for b in x]
def _left1(b): return tuple([b[0] - 1] + list(b[1:]))
def _right1(b): return tuple([b[0] + 1] + list(b[1:]))

class TowerContinuation(object):
    def __init__(self, x, w, h):
        self.x = x
        self.w = w
        self.h = h
    def __call__(self, k):
        return [(self.x,self.w,self.h)] + k

# name, dimensions
blocks = {  # "1x1": (1.,1.),
    #          "2x1": (2.,1.),
    #          "1x2": (1.,2.),
    "3x1": (3., 1.),
    "1x3": (1., 3.),
    #          "4x1": (4.,1.),
    #          "1x4": (1.,4.)
}
epsilon = 0.05

# Ensures axis aligned blocks


def xOffset(w, h):
    assert w == int(w)
    w = int(w)
    if w % 2 == 1:
        return 0.5
    return 0.
def _range(n):
    if n < 100: return range(n)
    raise ValueError()
def _fold(l): return lambda x0: lambda f: reduce(
    lambda a, x: f(x)(a), l[::-1], x0)

ttower = baseType("tower")
primitives = [
        Primitive("left", arrow(ttower, ttower), _left),
        Primitive("right", arrow(ttower, ttower), _right),
    ] + [Primitive(name, arrow(ttower,ttower), TowerContinuation(xOffset(w, h), w - 2*epsilon, h - epsilon))
         for name, (w, h) in blocks.items()] + [
                 Primitive("range", arrow(tint, tlist(tint)), _range),
                 Primitive("fold", arrow(tlist(t0), t1, arrow(t0, t1, t1), t1), _fold),
         ]
