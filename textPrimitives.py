from program import *
from makeTextTasks import delimiters

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
def _identity(x): return x

primitives = [
    Primitive("0",tint,0),
    Primitive("len",arrow(tstr,tint),len),
    Primitive("incr",arrow(tint,tint),_increment),
    Primitive("decr",arrow(tint,tint),_decrement),
    Primitive("empty",tstr,""),
    Primitive("caseLower",arrow(tstr,tstr), _lower),
    Primitive("caseUpper",arrow(tstr,tstr), _upper),
    Primitive("caseCapitalize",arrow(tstr,tstr), _capitalize),
    Primitive("++",arrow(tstr,tstr,tstr), _append),
    Primitive("slice", arrow(tint,tint,tstr,tstr),_slice),
    Primitive("nth", arrow(tint, tlist(tstr), tstr),_index),
    Primitive("map", arrow(arrow(tstr,tstr), tlist(tstr), tlist(tstr)),_map),
    #Primitive("find", arrow(tcharacter, tstr, tint),_find),
    #Primitive("replace", arrow(tstr, tstr, tstr, tstr),_replace),
    Primitive("split", arrow(tcharacter, tstr, tlist(tstr)),_split),
    Primitive("join", arrow(tstr, tlist(tstr), tstr),_join),
    Primitive("chr->str", arrow(tcharacter, tstr), _identity),
] + [ Primitive("'%s'"%d, tcharacter, d) for d in delimiters ]
