from __future__ import print_function

import random
import time
import traceback
import sys
import os
import subprocess
import math
import cPickle as pickle
from itertools import chain, imap


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def hashable(v):
    """Determine whether `v` can be hashed."""
    try:
        hash(v)
    except TypeError:
        return False
    return True


def flatten(x):
    """Recursively unroll iterables"""
    try:
        for e in chain(*imap(flatten, x)):
            yield e
    except TypeError: # not iterable
        yield x


NEGATIVEINFINITY = float('-inf')
POSITIVEINFINITY = float('inf')

PARALLELMAPDATA = None


def parallelMap(numberOfCPUs, f, *xs):
    global PARALLELMAPDATA

    if numberOfCPUs == 1: return map(f,*xs)

    n = len(xs[0])
    for x in xs: assert len(x) == n
    
    assert PARALLELMAPDATA is None
    PARALLELMAPDATA = (f,xs)

    from multiprocessing import Pool

    # Randomize the order in case easier ones come earlier or later
    permutation = range(n)
    random.shuffle(permutation)
    inversePermutation = dict(zip(permutation, range(n)))

    workers = Pool(numberOfCPUs)
    chunk = max(1,int(n/(numberOfCPUs*2)))
    ys = workers.map(parallelMapCallBack, permutation, chunksize = chunk)
    workers.terminate()

    PARALLELMAPDATA = None
    return [ ys[inversePermutation[j]] for j in range(n) ]


def parallelMapCallBack(j):
    global PARALLELMAPDATA
    f, xs = PARALLELMAPDATA
    try:
        return f(*[ x[j] for x in xs ])
    except Exception as e:
        eprint("Exception in worker during lightweight parallel map:\n%s"%(traceback.format_exc()))
        raise e


def log(x):
    t = type(x)
    if t == int or t == float:
        if x == 0: return NEGATIVEINFINITY
        return math.log(x)
    return x.log()


def exp(x):
    t = type(x)
    if t == int or t == float:
        return math.exp(x)
    return x.exp()


def lse(x,y = None):
    if y is None:
        largest = None
        if len(x) == 0: raise Exception('LSE: Empty sequence')
        if len(x) == 1: return x[0]
        # If these are just numbers...
        t = type(x[0])
        if t == int or t == float:
            largest = max(*x)
            return largest + math.log(sum(math.exp(z - largest) for z in x))
        # Must be torch
        return torchSoftMax(x)
    else:
        if x is NEGATIVEINFINITY: return y
        if y is NEGATIVEINFINITY: return x
        tx = type(x)
        ty = type(y)
        if (ty == int or ty == float) and (tx == int or tx == float):
            if x > y: return x + math.log(1. + math.exp(y - x))
            else: return y + math.log(1. + math.exp(x - y))
        return torchSoftMax(x,y)

def torchSoftMax(x,y = None):
    from torch.nn.functional import log_softmax
    import torch
    if y is None:
        if isinstance(x,list):
            x = torch.cat(x)
        return (x - log_softmax(x, dim = 0))[0]
    x = torch.cat((x,y))
    # this is so stupid
    return (x - log_softmax(x, dim = 0))[0]


def invalid(x):
    return math.isinf(x) or math.isnan(x)


def valid(x): return not invalid(x)


def callCompiled(f, *arguments, **keywordArguments):
    pypyArgs = []
    profile = keywordArguments.pop('profile', None)
    if profile:
        pypyArgs = ['-m', 'vmprof', '-o', profile]

    p = subprocess.Popen(['pypy'] + pypyArgs + ['compiledDriver.py'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    request = {
        "function": f,
        "arguments": arguments,
        "keywordArguments": keywordArguments,
    }
    start = time.time()
    pickle.dump(request, p.stdin)
    eprint("Wrote serialized message for {} in time {}".format(f.__name__, time.time() - start))

    success, result = pickle.load(p.stdout)
    eprint("Total pypy return time", time.time() - start)

    if not success:
        sys.exit(1)

    return result

class timing(object):
    def __init__(self,message):
        self.message = message
        
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        eprint("%s in %.1f seconds"%(self.message,
                                     time.time() - self.start))

def batches(data, size = 1):
    import random
    # Randomly permute the data
    data = list(data)
    random.shuffle(data)

    start = 0
    while start < len(data):
        yield data[start:size+start]
        start += size

def sampleDistribution(d):
    """
    Expects d to be a list of tuples
    The first element should be the probability
    If the tuples are of length 2 then it returns the second element
    Otherwise it returns the suffix tuple
    """
    import random
    
    z = float(sum(t[0] for t in d))
    r = random.random()
    u = 0.
    for t in d:
        p = t[0]/z
        if r < u + p:
            if len(t) <= 2: return t[1]
            else: return t[1:]
        u += p
    assert False

def testTrainSplit(x, trainingFraction, seed = 0):
    import random
    random.seed(seed)
    training = range(len(x))
    random.shuffle(training)
    training = set(training[:int(len(x)*trainingFraction)])

    train = [t for j,t in enumerate(x) if j in training ]
    test = [t for j,t in enumerate(x) if j not in training ]
    return test, train

def numberOfCPUs():
    import multiprocessing
    return multiprocessing.cpu_count()
    
    
def loadPickle(f):
    with open(f,'rb') as handle:
        d = pickle.load(handle)
    return d

def fst(l): return l[0]
