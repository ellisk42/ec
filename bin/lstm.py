import datetime
import os
import random
import math
from functools import reduce

import binutil  # required to import from dreamcoder modules

from dreamcoder.ec import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint, tlist, treal
from dreamcoder.utilities import numberOfCPUs
from dreamcoder.domains.list.listPrimitives import *
from dreamcoder.domains.list.main import retrieveJSONTasks, LearnedFeatureExtractor

# Primitives
def _incrf(x): return lambda x: x + 1.0
def _incrfeneg1(x): return lambda x: x + 0.1
def _incrfeneg2(x): return lambda x: x + 0.01
def _incrfeneg3(x): return lambda x: x + 0.001
def _incrfeneg4(x): return lambda x: x + 0.0001

def _incr(x): return lambda x: x + 1
def _decr(x): return lambda x: x - 1

def _addf(x): return lambda y: x + y
def _subf(x): return lambda y: x - y
def _multf(x): return lambda y: x * y

# inverse
def _inversef(x): return 1.0/x
# negate
def _negatef(x): return -x

# multiply and add
def _multbias(x): return lambda m: lambda a: m*x + a

# division by 2
def _div2(x): return x/2.0
# division by 10
def _div10(x): return x/10.0
# elementwise multiplication
def _elem_mult(l): return lambda m: [a*b for a,b in zip(l,m)]
# elementwise addition
def _elem_add(l): return lambda m: [a+b for a,b in zip(l,m)]
# 3 input elementwise addition
def _elem_add_3in(l): return lambda a1: lambda a2: [c+d for c,d in zip(a2, [a+b for a,b in zip(l,a1)])]
# create one hot encoding vector at desired index
def _ohei(l): return lambda n: [1 if n==i else 0 for i in range(len(l))]
# activations
def _sigmoid(x): return 1 / (1 + math.exp(-x))
def _tanh(x): return math.tanh(x)
def _relu(x): return max(0.0, x)

# remove last element from the list
def _cut_last(l): return l[:-1]

# get the last element from the list
def _get_last(l): return l[-1]

# truncations for rounding
def _trunc8f(x): return (math.trunc(1e8*x))*1.0e-8
def _trunc4f(x): return (math.trunc(1e4*x))*1.0e-4

# hardcode 9x1 list to 3x3 list of lists
def _list_to_matrix(l):
    row1 = l[0:3]
    row2 = l[3:6]
    row3 = l[6:9]
    return [row1, row2, row3]

# sum of list
def _sumf(l):
    sum = 0.0
    for el in l:
        sum += el
    return sum

# dot product of 2 lists
def _dot_product(l1): return lambda l2: np.dot(l1, l2)

# multiplying a list of lists with a list
def _matrix_mult(m): return lambda l: [_dot_product(m[0])(l), _dot_product(m[1])(l), _dot_product(m[2])(l)]

# multiplying a 9x1 "array" list with a 3x1 list
def _matrix_mult_list(m_list):
    m = _list_to_matrix(m_list)
    return lambda l: _matrix_mult(m)(l)

# define lstm cell and its components
def _forget_gate(l):
    htm1 = l[0:3]
    ctm1 = l[3:6]
    xt = l[6]
    # ft=sigmoid(Ufht−1+Wfxt+bf)
    #return lambda Uf: lambda Wf: lambda bf: list(map(_sigmoid, Uf @ htm1 + np.dot(Wf, xt) + bf))
    return lambda Uf: lambda Wf: lambda bf: list(map(_sigmoid, _elem_add_3in(_matrix_mult_list(Uf)(htm1))(np.dot(Wf, xt))(bf)))

def _input_gate(l):
    htm1 = l[0:3]
    ctm1 = l[3:6]
    xt = l[6]
    # it=sigmoid(Uiht−1+Wixt+bi)
    #return lambda Ui: lambda Wi: lambda bi: list(map(_sigmoid, Ui @ htm1 + np.dot(Wi, xt) + bi))
    return lambda Ui: lambda Wi: lambda bi: list(map(_sigmoid, _elem_add_3in(_matrix_mult_list(Ui)(htm1))(np.dot(Wi, xt))(bi)))

def _candidate_cell_state(l):
    htm1 = l[0:3]
    ctm1 = l[3:6]
    xt = l[6]
    # ctilda=tanh(Ucht−1+Wcxt+bc)
    #return lambda Uc: lambda Wc: lambda bc: list(map(_tanh, Uc @ htm1 + np.dot(Wc, xt) + bc))
    return lambda Uc: lambda Wc: lambda bc: list(map(_tanh, _elem_add_3in(_matrix_mult_list(Uc)(htm1))(np.dot(Wc, xt))(bc)))

def _output_gate(l):
    htm1 = l[0:3]
    ctm1 = l[3:6]
    xt = l[6]
    # ot=sigmoid(Uoht−1+Woxt+bo)
    #return lambda Uo: lambda Wo: lambda bo: list(map(_sigmoid, Uo @ htm1 + np.dot(Wo, xt) + bo))
    return lambda Uo: lambda Wo: lambda bo: list(map(_sigmoid, _elem_add_3in(_matrix_mult_list(Uo)(htm1))(np.dot(Wo, xt))(bo)))

def _new_cell_state(ft):
    # ct=ft∗ct−1+it∗ctilda
    return lambda ctm1: lambda it: lambda ctilda: _elem_add(_elem_mult(ft)(ctm1))(_elem_mult(it)(ctilda))

def _new_hidden_state(ot):
    # ht=ot∗tanh(ct)
    return lambda ct: _elem_mult(ot)(list(map(_tanh, ct)))

# lstm cell
# input is l = ht-1 | ct-1 | xt; output is ht | ct
# hardcoded feature size = 1, hidden units = 3
def _lstm_cell(l):
    htm1 = l[0:3]
    ctm1 = l[3:6]
    xt = l[6]

    return lambda Uf: lambda Wf: lambda bf: lambda Ui: lambda Wi: lambda bi: lambda Uc: lambda Wc: lambda bc: lambda Uo: lambda Wo: lambda bo: np.concatenate((_elem_mult(list(map(_sigmoid, _elem_add_3in(_matrix_mult_list(Uo)(htm1))(np.dot(Wo, xt))(bo))))(list(map(_tanh, _elem_add(_elem_mult(list(map(_sigmoid, _elem_add_3in(_matrix_mult_list(Uf)(htm1))(np.dot(Wf, xt))(bf))))(ctm1))(_elem_mult(list(map(_sigmoid, _elem_add_3in(_matrix_mult_list(Ui)(htm1))(np.dot(Wi, xt))(bi))))(list(map(_tanh, _elem_add_3in(_matrix_mult_list(Uc)(htm1))(np.dot(Wc, xt))(bc)))))))), _elem_add(_elem_mult(list(map(_sigmoid, _elem_add_3in(_matrix_mult_list(Uf)(htm1))(np.dot(Wf, xt))(bf))))(ctm1))(_elem_mult(list(map(_sigmoid, _elem_add_3in(_matrix_mult_list(Ui)(htm1))(np.dot(Wi, xt))(bi))))(list(map(_tanh, _elem_add_3in(_matrix_mult_list(Uc)(htm1))(np.dot(Wc, xt))(bc))))))).tolist()

# from listPrimitives
def _flatten(l): return [x for xs in l for x in xs]

def _range(n):
    if n < 100: return list(range(n))
    raise ValueError()

def _if(c): return lambda t: lambda f: t if c else f

def _negate(x): return -x

def _append(x): return lambda y: x + y

def _cons(x): return lambda y: [x] + y

def _car(x): return x[0]

def _cdr(x): return x[1:]

def _isEmpty(x): return x == []

def _single(x): return [x]

def _slice(x): return lambda y: lambda l: l[x:y]

def _map(f): return lambda l: list(map(f, l))

def _zip(a): return lambda b: lambda f: list(map(lambda x,y: f(x)(y), a, b))

def _mapi(f): return lambda l: list(map(lambda i_x: f(i_x[0])(i_x[1]), enumerate(l)))

def _reduce(f): return lambda x0: lambda l: reduce(lambda a, x: f(a)(x), l, x0)

def _reducei(f): return lambda x0: lambda l: reduce(
    lambda a, t: f(t[0])(a)(t[1]), enumerate(l), x0)

def _fold(l): return lambda x0: lambda f: reduce(
    lambda a, x: f(x)(a), l[::-1], x0)

def _eq(x): return lambda y: x == y

def _gt(x): return lambda y: x > y

def _index(j): return lambda l: l[j]

def _replace(f): return lambda lnew: lambda lin: _flatten(
    lnew if f(i)(x) else [x] for i, x in enumerate(lin))

def _filter(f): return lambda l: list(filter(f, l))

def _any(f): return lambda l: any(f(x) for x in l)

def _all(f): return lambda l: all(f(x) for x in l)

if __name__ == "__main__":

    # Options more or less copied from list.py

    args = commandlineArguments(
        enumerationTimeout=10, activation='tanh',
        iterations=10, recognitionTimeout=3600,
        a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
        helmholtzRatio=0.5, structurePenalty=1., #featureExtractor=LearnedFeatureExtractor, 
        CPUs=numberOfCPUs())

    timestamp = datetime.datetime.now().isoformat()
    outdir = 'experimentOutputs/demo/'
    os.makedirs(outdir, exist_ok=True)
    outprefix = outdir + timestamp
    args.update({"outputPrefix": outprefix})

    # Create new primitives
    new_primitives = [
        Primitive("incrf", arrow(treal, treal), _incrf),
        #Primitive("inversef", arrow(treal, treal), _inversef),
        #Primitive("negatef", arrow(treal, treal), _negatef),
        Primitive("incrfeneg1", arrow(treal, treal), _incrfeneg1),
        Primitive("incrfeneg2", arrow(treal, treal), _incrfeneg2),
        Primitive("incrfeneg3", arrow(treal, treal), _incrfeneg3),
        Primitive("incrfeneg4", arrow(treal, treal), _incrfeneg4),
        Primitive("addf", arrow(treal, treal, treal), _addf),
        #Primitive("+", arrow(tint, tint, tint), _addf),
        #Primitive("-", arrow(tint, tint, tint), _subf),
        #Primitive("incr", arrow(tint, tint), _incr),
        #Primitive("decr", arrow(tint, tint), _decr),
        #Primitive("subf", arrow(treal, treal, treal), _subf),
        Primitive("multf", arrow(treal, treal, treal), _multf),
        #Primitive("multbias", arrow(treal, treal, treal, treal), _multbias),
        Primitive("div2", arrow(treal, treal), _div2),
        Primitive("div10", arrow(treal, treal), _div10),
        #Primitive("elem_mult", arrow(tlist(t0), tlist(t0), tlist(t0)), _elem_mult),
        #Primitive("elem_add", arrow(tlist(t0), tlist(t0), tlist(t0)), _elem_add),
        #Primitive("elem_add_3in", arrow(tlist(t0), tlist(t0), tlist(t0), tlist(t0)), _elem_add_3in),
        #Primitive("ohei", arrow(tlist(t0), tint, tlist(t0)), _ohei),
        #Primitive("sigmoid", arrow(treal, treal), _sigmoid),
        #Primitive("tanh", arrow(treal, treal), _tanh),
        #Primitive("relu", arrow(treal, treal), _relu),
        #Primitive("trunc8f", arrow(treal, treal), _trunc8f),
        #Primitive("trunc4f", arrow(treal, treal), _trunc4f),
        #Primitive("cut_last", arrow(tlist(t0), tlist(t0)), _cut_last),
        #Primitive("get_last", arrow(tlist(t0), t0), _get_last),
        #Primitive("length", arrow(tlist(t0), tint), len),
        #Primitive("sumf", arrow(tlist(treal), treal), _sumf) 
        #Primitive("dot_product", arrow(tlist(treal), tlist(treal), treal), _dot_product), 
        #Primitive("matrix_mult_list", arrow(tlist(treal), tlist(treal), tlist(treal)), _matrix_mult_list),
        #Primitive("f_i_o_gate", arrow(tlist(treal), tlist(treal), tlist(treal), tlist(treal), tlist(treal)), _forget_gate),
        #Primitive("candidate_cell_state", arrow(tlist(treal), tlist(treal), tlist(treal), tlist(treal), tlist(treal)), _candidate_cell_state),
        #Primitive("new_cell_state", arrow(tlist(treal), tlist(treal), tlist(treal), tlist(treal), tlist(treal)), _new_cell_state),
        #Primitive("new_hidden_state", arrow(tlist(treal), tlist(treal), tlist(treal)), _new_hidden_state),
        #Primitive("lstm_cell", arrow(tlist(treal), tlist(treal), tlist(treal), tlist(treal), tlist(treal), tlist(treal), tlist(treal), tlist(treal), tlist(treal), tlist(treal), tlist(treal), tlist(treal), tlist(treal), tlist(treal)), _lstm_cell)
    ]

    """list_primitives = [
        #Primitive("reduce", arrow(arrow(t1, t0, t1), t1, tlist(t0), t1), _reduce)
        #Primitive("flatten", arrow(tlist(tlist(t0)), tlist(t0)), _flatten),
        #Primitive("sum", arrow(tlist(tint), tint), sum),
        #Primitive("all", arrow(arrow(t0, tbool), tlist(t0), tbool), _all),
        #Primitive("any", arrow(arrow(t0, tbool), tlist(t0), tbool), _any),
        #Primitive("replace", arrow(arrow(tint, t0, tbool), tlist(t0), tlist(t0), tlist(t0)), _replace)]"""

    list_primitives = [
        #Primitive(str(j), tint, j) for j in range(11)] + [
        Primitive(str(float(j)), treal, float(j)) for j in range(11)] + [
        Primitive(str(0.1), treal, 0.1),
        Primitive(str(0.01), treal, 0.01), 
        Primitive(str(0.001), treal, 0.001)
        #Primitive("empty", tlist(t0), []),
        #Primitive("singleton", arrow(t0, tlist(t0)), _single), 
        #Primitive("range", arrow(tint, tlist(tint)), _range),
        #Primitive("++", arrow(tlist(t0), tlist(t0), tlist(t0)), _append),
        #Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), _cons),
        #Primitive("car", arrow(tlist(t0), t0), _car),
        #Primitive("cdr", arrow(tlist(t0), tlist(t0)), _cdr),
        #Primitive("empty?", arrow(tlist(t0), tbool), _isEmpty), 
        #Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
        #Primitive("mapi", arrow(arrow(tint, t0, t1), tlist(t0), tlist(t1)), _mapi),
        #Primitive("reducei", arrow(arrow(tint, t1, t0, t1), t1, tlist(t0), t1), _reducei),
        #Primitive("zip", arrow(tlist(t0), tlist(t1), arrow(t0, t1, t2), tlist(t2)), _zip),
        #Primitive("fold", arrow(tlist(t0), t1, arrow(t0, t1, t1), t1), _fold),
        #Primitive("true", tbool, True),
        #Primitive("sort", arrow(tlist(tint), tlist(tint)), sorted),
        #Primitive("negate", arrow(tint, tint), _negate),
        #Primitive("eq?", arrow(tint, tint, tbool), _eq),
        #Primitive("gt?", arrow(tint, tint, tbool), _gt),
        #Primitive("index", arrow(tint, tlist(t0), t0), _index)]
        #Primitive("filter", arrow(arrow(t0, tbool), tlist(t0), tlist(t0)), _filter),
        #Primitive("slice", arrow(tint, tint, tlist(t0), tlist(t0)), _slice),
        #Primitive("if", arrow(tbool, t0, t0, t0), _if)
        ]

    # Reuse list primitives
    primitives = new_primitives + list_primitives

    # Create grammar
    grammar = Grammar.uniform(primitives)

    # Get training and testing tasks
    training = retrieveJSONTasks("data/lstm_training_incremental.json")
    testing = retrieveJSONTasks("data/lstm_toy_training.json")

    # EC iterate
    generator = ecIterator(grammar,
                           training,
                           testingTasks=testing,
                           **args)
    for i, _ in enumerate(generator):
        print('ecIterator count {}'.format(i))