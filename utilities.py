import math

def lse(x,y):
    if x > y: return x + math.log(1. + math.exp(y - x))
    else: return y + math.log(1. + math.exp(x - y))
