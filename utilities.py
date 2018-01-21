from __future__ import print_function

import time
import traceback
import sys
import os
import subprocess
import math
import cPickle as pickle


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


PARALLELMAPDATA = None
def parallelMap(numberOfCPUs, f, *xs):
    global PARALLELMAPDATA

    if numberOfCPUs == 1: return map(f,*xs)

    for x in xs: assert len(x) == len(xs[0])
    assert PARALLELMAPDATA is None
    PARALLELMAPDATA = (f,xs)
    
    from multiprocessing import Pool
    
    workers = Pool(numberOfCPUs)
    ys = workers.map(parallelMapCallBack, range(len(xs[0])))
    workers.terminate()

    PARALLELMAPDATA = None
    return ys
    
def parallelMapCallBack(j):
    global PARALLELMAPDATA
    (f,xs) = PARALLELMAPDATA
    try:
        return f(*[ x[j] for x in xs ])
    except Exception as e:
        eprint("Exception in worker during lightweight parallel map:\n%s"%(traceback.format_exc()))
        raise e


def log(x):
    t = type(x)
    if t == int or t == float:
        if t == 0: return NEGATIVEINFINITY
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

NEGATIVEINFINITY = float('-inf')
POSITIVEINFINITY = float('inf')

def invalid(x):
    return math.isinf(x) or math.isnan(x)
def valid(x): return not invalid(x)


def callCompiled(f, *arguments, **keywordArguments):
    # profile is a keyword argument for callCompiled, _not_ whatever
    # compiled function is being called
    profile = keywordArguments.get('profile',None)
    if profile in keywordArguments: del keywordArguments['profile']
    
    if profile is None: pythonArguments = []
    else: pythonArguments = ['-m','vmprof','-o',profile]
    
    p = subprocess.Popen(['pypy'] + pythonArguments + ['compiledDriver.py'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    request = {
        "function": f,
        "arguments": arguments,
        "keywordArguments": keywordArguments,
    }
    start = time.time()
    pickle.dump(request, p.stdin)
    eprint("Wrote serialized message in time", time.time() - start)

    (success, result) = pickle.load(p.stdout)
    eprint("Total pypy return time", time.time() - start)

    if not success:
        os.exit(1)

    return result
