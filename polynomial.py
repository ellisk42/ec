from ec import commandlineArguments, explorationCompression
from grammar import Grammar
from arithmeticPrimitives import addition, multiplication, k0, k1
from type import tint, arrow
from task import RegressionTask


primitives = [addition, multiplication, k0, k1]

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
    baseGrammar = Grammar.uniform(primitives)
    explorationCompression(baseGrammar, tasks,
                           outputPrefix = "experimentOutputs/polynomial",
                           **commandlineArguments(frontierSize = 10**4,
                                                  iterations = 10,
                                                  pseudoCounts = 10.0))
