from ec import *
from arithmeticPrimitives import *

polynomialPrimitives = [addition, multiplication, real]

MAXIMUMCOEFFICIENT = 9
NUMBEROFEXAMPLES = 5
tasks = [ DifferentiableTask("%dx^2 + %dx + %d"%(a,b,c),
                             arrow(tint,tint),
                             [((x,),a*x*x + b*x + c) for x in range(NUMBEROFEXAMPLES+1) ],
                             loss = squaredErrorLoss,
                             features = [float(a*x*x + b*x + c) for x in range(NUMBEROFEXAMPLES+1) ],
                             likelihoodThreshold = -0.5)
          for a in range(MAXIMUMCOEFFICIENT+1)
          for b in range(MAXIMUMCOEFFICIENT+1)
          for c in range(MAXIMUMCOEFFICIENT+1) ]

if __name__ == "__main__":
    explorationCompression(polynomialPrimitives, tasks,
                           outputPrefix = "experimentOutputs/continuousPolynomial",
                           **commandlineArguments(frontierSize = 10**2,
                                                  iterations = 5,
                                                  pseudoCounts = 10.0))
