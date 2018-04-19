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

specialCharacters = {' ': 'SPACE',
                     ')': 'RPAREN',
                     '(': 'LPAREN'}

primitives = [
    Primitive("char-eq?",arrow(tcharacter,tcharacter,tboolean),_eq)
] + [ Primitive("'%s'"%d, tcharacter, d) for d in delimiters if d not in specialCharacters] + \
[ Primitive(name, tcharacter, value) for value, name in specialCharacters.iteritems() ]
