from utilities import *

from differentiation import *

placeholders = []
def debug(t, clamp, lr, steps, attempts):
    global placeholders
    import re

    
    n = re.sub(r'([0-9])x',r'\1*x',
               t.name.replace('[','(').replace(']',')').replace('^','**').replace(')(',')*('))
    n = "lambda x: %s"%n
    # eprint(t)
    # eprint(n)
    f = eval(n)
    for (x,),y in t.examples:
        assert abs(f(x) - y) < 0.1, \
            "Expected y = %f, but instead I got %f"%(y,f(x))

    placeholders = []
    def replace(m):
        placeholders.append(Placeholder(normal()))
        return "placeholders[%d]"%(len(placeholders) - 1)
    n = re.sub(r'[0-9]+\.[0-9]', replace, n)
    # eprint(n)

    f = eval(n)
    def _clamp(z):
        L=15.
        if clamp:
            return z.clamp(l=-L,u=L)
        return z
    
    l = sum((_clamp(f(x)) - y).square() for (x,),y in t.examples )/len(t.examples)

    return l.restartingOptimize(placeholders,
                                attempts=attempts,
                                lr=lr,
                                steps=steps)

def debugMany(ts, clamp, lr, steps, attempts):
    return parallelMap(numberOfCPUs()/2,lambda t: debug(t,clamp, lr, steps, attempts), ts)
