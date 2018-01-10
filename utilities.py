import math

def lse(x,y = None):
    if y == None:
        largest = None
        if len(x) == 0: raise Exception('LSE: Empty sequence')
        if len(x) == 1: return x[0]
        for z in x:
            if largest == None or z > largest: largest = z
        return largest + math.log(sum(math.exp(z - largest) for z in x))
    else:
        if x > y: return x + math.log(1. + math.exp(y - x))
        else: return y + math.log(1. + math.exp(x - y))

NEGATIVEINFINITY = float('-inf')
