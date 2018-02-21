from program import *

from arithmeticPrimitives import *
from logicalPrimitives import *

import tuplePrimitives

def _concatenate(x): return lambda y: x+y
def _single(x): return [x]
def _map(f): return lambda l: map(f,l)
def _negation(x): return -x
def _left(x): return [(z - 1,o) for z,o in x ]
def _right(x): return [(z + 1,o) for z,o in x ]

ttower = tlist(tpair(tint,tbool))

primitives = [
              Primitive("++", arrow(tlist(t0),tlist(t0),tlist(t0)), _concatenate),
              Primitive("horizontal", ttower, [(0,True)]),
              Primitive("vertical", ttower, [(0,False)]),
              Primitive("left", arrow(ttower,ttower), _left),
              Primitive("right", arrow(ttower,ttower), _right),
    #              Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
              #primitiveNot,
              #Primitive("negate", arrow(tint,tint), _negation)
]
