from program import *

from arithmeticPrimitives import *
from logicalPrimitives import *

import tuplePrimitives

def _concatenate(x): return lambda y: x+y
def _single(x): return [x]
def _map(f): return lambda l: map(f,l)
def _negation(x): return -x
def _left(x): return map(lambda b: tuple([b[0] - 1] + list(b[1:])), x)
def _right(x): return map(lambda b: tuple([b[0] + 1] + list(b[1:])), x)

ttower = baseType("tower")

# name, dimensions
blocks = {#"1x1": (1.,1.),
#          "2x1": (2.,1.),
#          "1x2": (1.,2.),
          "3x1": (3.,1.),
          "1x3": (1.,3.),
          "4x1": (4.,1.),
          "1x4": (1.,4.)}
epsilon = 0.05

# Ensures axis aligned blocks
def xOffset(w,h):
    assert w == int(w)
    w = int(w)
    if w%2 == 1: return 0.5
    return 0.

primitives = [
              Primitive("do", arrow(ttower,ttower,ttower), _concatenate),
              Primitive("left", arrow(ttower,ttower), _left),
              Primitive("right", arrow(ttower,ttower), _right),
] + [ Primitive(name, ttower, [(xOffset(w,h), w - epsilon, h - epsilon)])
      for name, (w,h) in blocks.iteritems() ]
