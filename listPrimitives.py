from program import Primitive, Program
from grammar import Grammar
from type import tlist, tint, tbool, arrow

import math

def _addition(x): return lambda y: x + y
def _multiplication(x): return lambda y: x*y
def _negate(x): return -x
def _reverse(x): return list(reversed(x))
def _append(x): return lambda y: x + y
def _single(x): return [x]
def _slice(x): return lambda y: lambda l: l[x:y]
def _map(f): return lambda l: map(f, l)
def _reduce(f): return lambda x0: lambda l: reduce(lambda a, b: f(a)(b), l, x0)
def _filter(f): return lambda l: filter(f, l)
def _any(f): return lambda l: any(f(x) for x in l)
def _all(f): return lambda l: all(f(x) for x in l)
def _eq(x): return lambda y: x == y
def _mod(x): return lambda y: x%y
def _not(x): return not x
def _gt(x): return lambda y: x > y
def _index(j): return lambda l: l[j]
def _replace(i): return lambda lo: lambda ln: lo[:i]+ln+lo[i:]
def _find(x):
    def _inner(l):
        try:
            return l.index(x)
        except ValueError:
            return -1
    return _inner
def _isPrime(n):
    return n in {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199}
def _isSquare(n):
    return int(math.sqrt(n)) ** 2 == n


baseGrammar = Grammar.fromProductions([
    (-1.0, Primitive("index", arrow(tint, tlist(tint), tint), _index)),
    (-1.0, Primitive("+", arrow(tint, tint, tint), _addition)),
    (-1.0, Primitive("negate", arrow(tint, tint), _negate)),
    (-1.0, Primitive("*", arrow(tint, tint, tint), _multiplication)),
    (-1.0, Primitive("mod", arrow(tint, tint, tint), _mod)),
    (-1.0, Primitive("sort", arrow(tlist(tint), tlist(tint)), sorted)),
    (-1.0, Primitive("reverse", arrow(tlist(tint), tlist(tint)), _reverse)),
    (-1.0, Primitive("++", arrow(tlist(tint), tlist(tint), tlist(tint)), _append)),
    (-1.0, Primitive("singleton", arrow(tint, tlist(tint)), _single)),
    (-1.0, Primitive("slice", arrow(tint, tint, tlist(tint), tlist(tint)), _slice)),
    (-1.0, Primitive("len", arrow(tlist(tint), tint), len)),
    ( 0.0, Primitive("map", arrow(arrow(tint, tint), tlist(tint), tlist(tint)), _map)),
    ( 0.0, Primitive("reduce", arrow(arrow(tint, tint), tint, tlist(tint), tint), _reduce)),
    ( 0.0, Primitive("filter", arrow(arrow(tint, tbool), tlist(tint), tlist(tint)), _filter)),
    ( 0.0, Primitive("any", arrow(arrow(tint, tbool), tlist(tint), tbool), _any)),
    ( 0.0, Primitive("all", arrow(arrow(tint, tbool), tlist(tint), tbool), _all)),
    (-1.0, Primitive("eq?", arrow(tint, tint, tbool), _eq)),
    ( 0.0, Primitive("not", arrow(tbool, tbool), _not)),
    (-3.0, Primitive("gt?", arrow(tint, tint, tbool), _gt)),
    ( 0.0, Primitive("find", arrow(tint, tlist(tint), tint), _find)),
    ( 0.0, Primitive("replace", arrow(tint, tlist(tint), tlist(tint), tlist(tint)), _replace)),
    (-4.0, Primitive("is-prime", arrow(tint, tbool), _isPrime)),
    (-4.0, Primitive("is-square", arrow(tint, tbool), _isSquare)),
] + [ (-0.2, Primitive(str(j), tint, j)) for j in range(-2, 10) ])

primitives = baseGrammar.primitives
