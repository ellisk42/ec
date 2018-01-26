from program import Primitive, Program
from grammar import Grammar
from type import tlist, tint, tbool, arrow, t0, t1

import math

def _if(c): return lambda f: lambda t: t if c else f
def _and(x): return lambda y: x and y
def _or(x): return lambda y: x or y
def _addition(x): return lambda y: x + y
def _multiplication(x): return lambda y: x*y
def _negate(x): return -x
def _reverse(x): return list(reversed(x))
def _append(x): return lambda y: x + y
def _single(x): return [x]
def _slice(x): return lambda y: lambda l: l[x:y]
def _map(f): return lambda l: map(f, l)
def _reduce(f): return lambda x0: lambda l: reduce(lambda a, b: f(a)(b), l, x0)
def _eq(x): return lambda y: x == y
def _mod(x): return lambda y: x%y
def _not(x): return not x
def _gt(x): return lambda y: x > y
def _index(j): return lambda l: l[j]
def _replace(i): return lambda lo: lambda ln: lo[:i]+ln+lo[i:]
def _isPrime(n):
    return n in {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199}
def _isSquare(n):
    return int(math.sqrt(n)) ** 2 == n


primitives = [
    Primitive("singleton", arrow(t0, tlist(t0)), _single),
    Primitive("range", arrow(tint, tlist(tint)), range),
    Primitive("++", arrow(tlist(t0), tlist(t0), tlist(t0)), _append),
    Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
    Primitive("reduce", arrow(arrow(t1, t0, t1), t1, tlist(t0), t1), _reduce),

    Primitive("index", arrow(tint, tlist(t0), t0), _index),
    Primitive("slice", arrow(tint, tint, tlist(t0), tlist(t0)), _slice),
    Primitive("replace", arrow(tint, tlist(t0), tlist(t0), tlist(t0)), _replace),
    Primitive("reverse", arrow(tlist(t0), tlist(t0)), _reverse),

    Primitive("sort", arrow(tlist(tint), tlist(tint)), sorted),
    Primitive("+", arrow(tint, tint, tint), _addition),
    Primitive("negate", arrow(tint, tint), _negate),
    Primitive("*", arrow(tint, tint, tint), _multiplication),
    Primitive("mod", arrow(tint, tint, tint), _mod),
    Primitive("eq?", arrow(tint, tint, tbool), _eq),
    Primitive("gt?", arrow(tint, tint, tbool), _gt),
    Primitive("is-prime", arrow(tint, tbool), _isPrime),
    Primitive("is-square", arrow(tint, tbool), _isSquare),

    Primitive("if", arrow(tbool, t0, t0, t0), _if),
    Primitive("and", arrow(tbool, tbool, tbool), _and),
    Primitive("or", arrow(tbool, tbool, tbool), _or),
    Primitive("true", tbool, True),
    Primitive("not", arrow(tbool, tbool), _not),
] + [ Primitive(str(j), tint, j) for j in range(10) ]
