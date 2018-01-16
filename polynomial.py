from ec import *
from arithmeticPrimitives import *

polynomialPrimitives = [addition, multiplication,k0,k1]

MAXIMUMCOEFFICIENT = 9
NUMBEROFEXAMPLES = 5
tasks = [
    RegressionTask("%dx^2 + %dx + %d"%(a,b,c),
                   arrow(tint,tint),
                   [((x,), a*x*x + b*x + c) for x in range(NUMBEROFEXAMPLES+1) ],
                   features = [float(a*x*x + b*x + c) for x in range(NUMBEROFEXAMPLES+1) ],
                   cache = True)
          for a in range(MAXIMUMCOEFFICIENT+1)
          for b in range(MAXIMUMCOEFFICIENT+1)
          for c in range(MAXIMUMCOEFFICIENT+1)
]

if __name__ == "__main__":
    # import cPickle as pickle
    # print pickle.dumps(addition)
    # assert False
    explorationCompression(polynomialPrimitives, tasks,
                           **commandlineArguments(frontierSize = 10**4,
                                                  iterations = 5,
                                                  pseudoCounts = 10.0))
