from program import *
from type import *

def _addition(x): return lambda y: x + y
def _subtraction(x): return lambda y: x - y
subtraction = Primitive("-",
                        arrow(tint,arrow(tint,tint)),
                        _subtraction)
addition = Primitive("+",
                     arrow(tint,arrow(tint,tint)),
                     _addition)
def _multiplication(x): return lambda y: x*y
multiplication = Primitive("*",
                           arrow(tint,arrow(tint,tint)),
                           _multiplication)
k1 = Primitive("1",tint,1)
k_negative1 = Primitive("negative_1",tint,-1)
k0 = Primitive("0",tint,0)
real = Primitive("REAL",tint,None)
