from program import *

def _pair(x): return lambda y: (x,y)
def _first(x): return x[0]
def _second(x): return x[1]

primitives = [Primitive("pair", arrow(t0,t1,tpair(t0,t1)),_pair),
              Primitive("fst", arrow(tpair(t0,t1),t0),_first),
              Primitive("snd", arrow(tpair(t0,t1),t1),_second)
]
