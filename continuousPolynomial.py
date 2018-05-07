from ec import explorationCompression, commandlineArguments, Program
from grammar import Grammar
from arithmeticPrimitives import addition, multiplication, real
from task import DifferentiableTask, squaredErrorLoss, l1loss, Task
from type import tint, arrow
from utilities import *
from program import *

primitives = [addition, multiplication, real]

MAXIMUMCOEFFICIENT = 9
NUMBEROFEXAMPLES = 3
EXAMPLES = list(range(-(NUMBEROFEXAMPLES/2),
                 (NUMBEROFEXAMPLES - NUMBEROFEXAMPLES/2)))
tasks = [ DifferentiableTask("%dx^2 + %dx + %d"%(a,b,c),
                             arrow(tint,tint),
                             [((x,),a*x*x + b*x + c) for x in EXAMPLES ],
                             loss=squaredErrorLoss,
                             features=[float(a*x*x + b*x + c) for x in EXAMPLES ],
                             likelihoodThreshold=-0.1)
          for a in range(MAXIMUMCOEFFICIENT+1)
          for b in range(MAXIMUMCOEFFICIENT+1)
          for c in range(MAXIMUMCOEFFICIENT+1) ]

def makeFeatureExtractor(xxx_todo_changeme):
    (averages, deviations) = xxx_todo_changeme
    def featureExtractor(program, tp):
        e = program.visit(RandomParameterization.single)
        f = e.evaluate([])
        outputs = [float(f(x)) for x in EXAMPLES]
        features = Task.standardizeFeatures(averages, deviations, outputs)
        # eprint("program %s instantiates to %s; has outputs %s ; features %s"%(program, e, outputs, features))
        return features
    return featureExtractor

class RandomParameterization(object):
    def primitive(self, e):
        if e.name == 'REAL':
            v = random.randrange(MAXIMUMCOEFFICIENT+1)
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
    statistics = Task.standardizeTasks(tasks)
    featureExtractor = makeFeatureExtractor(statistics)
    
    explorationCompression(baseGrammar, tasks,
                           outputPrefix="experimentOutputs/continuousPolynomial",
                           **commandlineArguments(frontierSize=10**2,
                                                  iterations=5,
                                                  featureExtractor=featureExtractor,
                                                  pseudoCounts=10.0))
