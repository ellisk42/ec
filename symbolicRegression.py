from ec import explorationCompression, commandlineArguments, Program
from grammar import Grammar
from arithmeticPrimitives import addition, multiplication, real
from task import DifferentiableTask, squaredErrorLoss, l1loss, RegressionTask
from type import tint, arrow
from utilities import *
from program import *

primitives = [addition, multiplication, real]

MAXIMUMCOEFFICIENT = 5
NUMBEROFEXAMPLES = 5
EXAMPLERANGE = 1.
EXAMPLES = [ -EXAMPLERANGE + j*(2*EXAMPLERANGE/(NUMBEROFEXAMPLES-1))
             for j in range(NUMBEROFEXAMPLES) ]
COEFFICIENTS = range(-(MAXIMUMCOEFFICIENT/2),
                     (MAXIMUMCOEFFICIENT - MAXIMUMCOEFFICIENT/2))
tasks = [ DifferentiableTask("%dx^4 + %dx^3 + %dx^2 + %dx + %d"%(a,b,c,d,e),
                             arrow(tint,tint),
                             [((x,),a*x*x*x*x + b*x*x*x + c*x*x + d*x + e) for x in EXAMPLES ],
                             loss = squaredErrorLoss,
                             features = [float(a*x*x*x*x + b*x*x*x + c*x*x + d*x + e) for x in EXAMPLES ],
                             likelihoodThreshold = -0.1)
          for a in COEFFICIENTS
          for b in COEFFICIENTS
          for c in COEFFICIENTS
          for d in COEFFICIENTS
          for e in COEFFICIENTS]

def makeFeatureExtractor((averages, deviations)):
    def featureExtractor(program, tp):
        e = program.visit(RandomParameterization.single)
        f = e.evaluate([])
        outputs = [float(f(x)) for x in EXAMPLES]
        features = RegressionTask.standardizeFeatures(averages, deviations, outputs)
        # eprint("program %s instantiates to %s; has outputs %s ; features %s"%(program, e, outputs, features))
        return features
    return featureExtractor

class RandomParameterization(object):
    def primitive(self, e):
        if e.name == 'REAL':
            v = random.choice(COEFFICIENTS)
            return Primitive(str(v), e.tp, v)
        return e
    def invented(self,e): return e.body.visit(self)
    def abstraction(self,e): return Abstraction(e.body.visit(self))
    def application(self,e):
        return Application(e.f.visit(self),e.x.visit(self))
    def index(self,e): return e
RandomParameterization.single = RandomParameterization()
    
if __name__ == "__main__":
    baseGrammar = Grammar.uniform(primitives)
    statistics = RegressionTask.standardizeTasks(tasks)
    featureExtractor = makeFeatureExtractor(statistics)

    
    explorationCompression(baseGrammar, tasks,
                           outputPrefix = "experimentOutputs/regression",
                           **commandlineArguments(frontierSize = 10**2,
                                                  iterations = 5,
                                                  CPUs = numberOfCPUs(),
                                                  featureExtractor = featureExtractor,
                                                  pseudoCounts = 10.0))
