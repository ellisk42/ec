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
#def _reverse(x): return x[::-1]
def _strip(x): return x.strip()
def _eq(x): return lambda y: x == y

primitives = [
    # Primitive("0",tint,0),
    #Primitive("len",arrow(tstr,tint),len),
    # Primitive("+1",arrow(tint,tint),_increment),
    # Primitive("-1",arrow(tint,tint),_decrement),
    # Primitive("emptyString",tstr,""),
    Primitive("char-eq?",arrow(tcharacter,tcharacter,tboolean),_eq),
    Primitive("caseLower",arrow(tcharacter,tcharacter), _lower),
    Primitive("caseUpper",arrow(tcharacter,tcharacter), _upper),
    #Primitive("caseCapitalize",arrow(tstr,tstr), _capitalize),
    # Primitive("concatenate",arrow(tstr,tstr,tstr), _append),
    # Primitive("slice-string", arrow(tint,tint,tstr,tstr),_slice),
    # Primitive("nth", arrow(tint, tlist(tstr), tstr),_index),
    # Primitive("map-string", arrow(arrow(tstr,tstr), tlist(tstr), tlist(tstr)),_map),
    #Primitive("find", arrow(tcharacter, tstr, tint),_find),
    #Primitive("replace", arrow(tstr, tstr, tstr, tstr),_replace),
    # Primitive("strip", arrow(tstr,tstr),_strip),
    # Primitive("split", arrow(tcharacter, tstr, tlist(tstr)),_split),
    # Primitive("join", arrow(tstr, tlist(tstr), tstr),_join),
    # Primitive("chr2str", arrow(tcharacter, tstr), _identity),
] + [ Primitive("'%s'"%d, tcharacter, d) for d in delimiters if d != ' '] + \
[ Primitive("SPACE", tcharacter, ' ')]
