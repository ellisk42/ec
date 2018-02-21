from program import *

from arithmeticPrimitives import *
from logicalPrimitives import *

import tuplePrimitives

def _concatenate(x): return lambda y: x+y
def _single(x): return [x]
def _map(f): return lambda l: map(f,l)
def _negation(x): return -x

primitives = [k0,k1,k_negative1,
              addition,subtraction,
              Primitive("++", arrow(tlist(t0),tlist(t0),tlist(t0)), _concatenate),
              Primitive("singleton", arrow(t0, tlist(t0)), _single),
              Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
              primitiveTrue, primitiveFalse,
              primitiveNot,
              Primitive("negate", arrow(tint,tint), _negation)
] + tuplePrimitives.primitives
