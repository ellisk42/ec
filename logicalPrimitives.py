from program import *

def _nand(x): return lambda y: not (x and y)
primitive_nand = Primitive("nand",arrow(tbool,tbool,tbool), _nand)

primitiveTrue = Primitive("true", tbool, True)
primitiveFalse = Primitive("false", tbool, False)

def _not(x): return not x
primitiveNot = Primitive("not", arrow(tbool,tbool), _not)
