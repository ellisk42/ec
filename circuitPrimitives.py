from program import *

def _nand(x): return lambda y: not (x and y)
def _bind(x): return lambda body: body(x) 

primitives = [Primitive("nand",arrow(tbool,tbool,tbool), _nand)]

