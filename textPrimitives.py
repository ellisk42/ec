from program import *

def _increment(x): return x + 1
def _decrement(x): return x - 1
def _lower(x): return x.lower()
def _upper(x): return x.upper()
def _capitalize(x): return x.capitalize()
def _append(x): return lambda y: x + y
def _slice(x): return lambda y: lambda s: s[x:y]
def _index(n): return lambda x: x[n]
def _map(f): return lambda x: map(f,x)
def _find(pattern): return lambda s: s.index(pattern)
def _replace(original): return lambda replacement: lambda target: target.replace(original, replacement)
def _split(delimiter): return lambda s: s.split(delimiter)
def _join(delimiter): return lambda ss: delimiter.join(ss)

primitives = [
    Primitive("0",tint,0),
    Primitive("len",arrow(tstring,tint),len),
    Primitive("incr",arrow(tint,tint),_increment),
    Primitive("decr",arrow(tint,tint),_decrement),
    Primitive("emptyString",tstring,""),
    Primitive("lowercase",arrow(tstring,tstring), _lower),
    Primitive("uppercase",arrow(tstring,tstring), _upper),
    Primitive("capitalize",arrow(tstring,tstring), _capitalize),
    Primitive("++",arrow(tstring,tstring,tstring), _append),
    Primitive("','", tstring, ","),
    Primitive("' '", tstring, " "),
    Primitive("'<'", tstring, "<"),
    Primitive("'>'", tstring, ">"),
    Primitive("'.'", tstring, "."),
    Primitive("'@'", tstring, "@"),
    Primitive("slice", arrow(tint,tint,tstring,tstring),_slice),
    Primitive("nth", arrow(tint, tlist(tstring), tstring),_index),
    Primitive("map", arrow(arrow(tstring,tstring), tlist(tstring), tlist(tstring)),_map),
    Primitive("find", arrow(tstring, tstring, tint),_find),
    #Primitive("replace", arrow(tstring, tstring, tstring, tstring),_replace),
    Primitive("split", arrow(tstring, tstring, tlist(tstring)),_split),
    Primitive("join", arrow(tstring, tlist(tstring), tstring),_join)
]
