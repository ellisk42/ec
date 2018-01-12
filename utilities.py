import dill
import sys
import os
import math
import cPickle as pickle

def log(x):
    t = type(x)
    if t == int or t == float:
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
            for z in x:
                if largest == None or z > largest: largest = z
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
        return (x - log_softmax(x))[0]
    x = torch.cat((x,y))
    # this is so stupid
    return (x - log_softmax(x))[0]

NEGATIVEINFINITY = float('-inf')

def callCompiled(f, *arguments, **keywordArguments):
    modulePath = f.__module__

    ra,wa = os.pipe()
    rr,wr = os.pipe()
    p = os.fork()
    
    if p == 0:
        # Child
        os.close(wa)
        os.close(rr)

        pypy = '/usr/bin/pypy'
        os.execl(pypy,
                 pypy,'compiledDriver.py',str(ra),str(wr))
    else:
        # Parent
        os.close(ra)
        os.close(wr)
        serialized = dill.dumps({"arguments": arguments,
                                   "keywordArguments": keywordArguments,
                                   "function": f,
                                   "functionName": f.__name__,
                                   #"openModules": openModules,
                                   "module": modulePath})
        w = os.fdopen(wa,'wb')
        w.write(serialized)
        w.close()

        r = os.fdopen(rr,'rb')
        (success,returnValue) = dill.loads(r.read())

        if not success:
            print "Exception thrown in pypy process:"
            print returnValue
            assert False

        return returnValue
        
        
