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
blocks = {"unit": (1.,1.),
          "horizontalBrick": (2.,1.),
          "verticalBrick": (1.,2.),
          "wideHorizontal": (4.,1.),
          "tallVertical": (1.,4.)}
epsilon = 0.05

primitives = [
              Primitive("do", arrow(ttower,ttower,ttower), _concatenate),
              Primitive("left", arrow(ttower,ttower), _left),
              Primitive("right", arrow(ttower,ttower), _right),
] + [ Primitive(name, ttower, [(0., w - epsilon, h - epsilon)])
      for name, (w,h) in blocks.iteritems() ]
