from ec import *
from arithmeticPrimitives import *

polynomialPrimitives = [addition, multiplication, real]

def polynomialTask(a,b,c):
    return DifferentiableSSETask("%dx^2 + %dx + %d"%(a,b,c),
                                 arrow(tint,tint),
                                 [((x,),a*x*x + b*x + c) for x in range(NUMBEROFEXAMPLES+1) ],
                                 features = [float(a*x*x + b*x + c) for x in range(NUMBEROFEXAMPLES+1) ],
                                 likelihoodThreshold = -0.5)


MAXIMUMCOEFFICIENT = 9
NUMBEROFEXAMPLES = 5
tasks = [ polynomialTask(a,b,c)
          for a in range(MAXIMUMCOEFFICIENT+1)
          for b in range(MAXIMUMCOEFFICIENT+1)
          for c in range(MAXIMUMCOEFFICIENT+1) ]

if __name__ == "__main__":
    explorationCompression(polynomialPrimitives, tasks,
                           **commandlineArguments(frontierSize = 10**2,
                                                  iterations = 5,
                                                  pseudoCounts = 10.0))
