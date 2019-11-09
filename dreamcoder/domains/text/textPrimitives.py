from dreamcoder.program import *
from dreamcoder.domains.text.makeTextTasks import delimiters

def _isUpper(x): return x.isupper()

def _increment(x): return x + 1


def _decrement(x): return x - 1


def _lower(x): return x.lower()


def _upper(x): return x.upper()


def _capitalize(x): return x.capitalize()


def _append(x): return lambda y: x + y


def _slice(x): return lambda y: lambda s: s[x:y]


def _index(n): return lambda x: x[n]


def _map(f): return lambda x: list(map(f, x))


def _find(pattern): return lambda s: s.index(pattern)


def _replace(original): return lambda replacement: lambda target: target.replace(
    original, replacement)


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
    Primitive("char-eq?", arrow(tcharacter, tcharacter, tboolean), _eq),
    Primitive("STRING", tstr, None)
] + [Primitive("'%s'" % d, tcharacter, d) for d in delimiters if d not in specialCharacters] + \
    [Primitive(name, tcharacter, value) for value, name in specialCharacters.items()]


def _cons(x): return lambda y: [x] + y


def _car(x): return x[0]


def _cdr(x): return x[1:]


targetTextPrimitives = [
    Primitive("take-word", arrow(tcharacter, tstr, tstr), None),
    Primitive("drop-word", arrow(tcharacter, tstr, tstr), None),
    Primitive("append", arrow(tlist(t0), tlist(t0), tlist(t0)), None),
    Primitive("abbreviate", arrow(tstr, tstr), None),
    Primitive("last-word", arrow(tcharacter, tstr, tstr), None),
    Primitive("replace-character", arrow(tcharacter, tcharacter, tstr, tstr), None),
] + primitives + [
    Primitive("empty", tlist(t0), []),
    Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), _cons),
    Primitive("car", arrow(tlist(t0), t0), _car),
    Primitive("cdr", arrow(tlist(t0), tlist(t0)), _cdr)]
