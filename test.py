
from utilities import *

def f(x):
    if x <= 2:
        return 1
    return f(x - 1) + f(x - 2)

if __name__ == "__main__":
    parallelMap(2,f,[42,42])


