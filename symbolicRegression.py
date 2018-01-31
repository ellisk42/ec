from ec import explorationCompression, commandlineArguments, Program
from grammar import Grammar
from arithmeticPrimitives import addition, multiplication, real
from task import DifferentiableTask, squaredErrorLoss, l1loss, RegressionTask
from type import tint, arrow
from utilities import *
from program import *

primitives = [addition, multiplication, real]

MAXIMUMCOEFFICIENT = 7
NUMBEROFEXAMPLES = 5
EXAMPLERANGE = 2.
EXAMPLES = [ -EXAMPLERANGE + j*(2*EXAMPLERANGE/(NUMBEROFEXAMPLES-1))
             for j in range(NUMBEROFEXAMPLES) ]
COEFFICIENTS = range(-(MAXIMUMCOEFFICIENT/2),
                     (MAXIMUMCOEFFICIENT - MAXIMUMCOEFFICIENT/2))
def sign(n): return ['+','-'][int(n < 0)]
tasks = [ DifferentiableTask("%s%dx^4 %s %dx^3 %s %dx^2 %s %dx %s %d"%(" " if a >= 0 else "",a,
                                                                       sign(b),abs(b),
                                                                       sign(c),abs(c),
                                                                       sign(d),abs(d),
                                                                       sign(e),abs(e)),
                             arrow(tint,tint),
                             [((x,),a*x*x*x*x + b*x*x*x + c*x*x + d*x + e) for x in EXAMPLES ],
                             loss = squaredErrorLoss,
                             features = [float(a*x*x*x*x + b*x*x*x + c*x*x + d*x + e) for x in EXAMPLES ],
                             likelihoodThreshold = -0.3)
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
    
    test, train = testTrainSplit(tasks, 500/float(len(tasks)))

    # e = Program.parse("""(lambda (+ REAL
    # (* $0 (+ REAL
    # (* $0 (+ REAL
    # (* $0 (+ REAL 
    # (* $0 REAL)))))))))""")
    # eprint(e)
    # from fragmentGrammar import *
    # f = FragmentGrammar.uniform(baseGrammar.primitives + [Program.parse("(+ REAL $0)")])
    
    # eprint(f.closedLogLikelihood(arrow(tint,tint),e))
    # random.shuffle(tasks)
    # biggest = POSITIVEINFINITY
    # for t in train:
    #     l = t.logLikelihood(e)
    #     eprint(t, l)
    #     biggest = min(biggest,l)
    # eprint(biggest)
    # assert False
    
    explorationCompression(baseGrammar, train,
                           outputPrefix = "experimentOutputs/regression",
                           **commandlineArguments(frontierSize = 10**2,
                                                  iterations = 10,
                                                  CPUs = numberOfCPUs(),
                                                  structurePenalty = 5.,
                                                  featureExtractor = featureExtractor,
                                                  pseudoCounts = 10.0))
