from program import *

def _nand(x): return lambda y: not (x and y)

primitives = [Primitive("nand",arrow(tbool,tbool,tbool), _nand)]
