from ec import explorationCompression, commandlineArguments
from grammar import Grammar
from utilities import eprint, sampleDistribution
from circuitPrimitives import primitives
from task import RegressionTask
from type import arrow, tbool
from recognition import *

import itertools
import random

NUMBEROFTASKS = 10**3
inputDistribution = [#(1,1),
                     #(2,2),
                     (3,3),
                     (4,4),
                     (4,5),
    (4,6),
#    (4,5),
#    (4,6),
#    (4,7)
                     ]
MAXIMUMINPUTS = max(i for p,i in inputDistribution)
gateDistribution = [(1,1),
                    (2,2),
#                    (2,3),
#                    (4,4),
                    #(5,5),
                    #(6,5),
]
operationDistribution = [(1,'NOT'),
                         (2,'AND'),
                         (2,'OR'),
                         (3,'m2'),
                         (2,'m4')]

INPUTSPERGATE = {'NOT': 1,
                 'AND': 2,
                 'OR': 2,
                 'm2': 3,
                 'm4': 6}
GATEFUN = {'NOT': lambda x: not x,
           'AND': lambda x,y: x and y ,
           'OR': lambda x,y: x or y,
           'm2': lambda x,y,c: [x,y][int(c)],
           'm4': lambda a,b,c,d,x,y: [ [a,b][int(x)], [c,d][int(x)] ][int(y)]}


class Circuit(object):
    def __init__(self, _ = None, numberOfInputs = None, numberOfGates = None):
        assert numberOfInputs != None
        assert numberOfGates != None

        self.numberOfInputs = numberOfInputs

        self.operations = []
        while not self.isConnected():
            self.operations = []
            while len(self.operations) < numberOfGates:
                validInputs = range(-numberOfInputs, len(self.operations))
                gate = sampleDistribution(operationDistribution)
                if 'm' in gate:
                    if self.numberOfInputs < INPUTSPERGATE[gate]: continue
                    arguments = list(np.random.choice(validInputs,
                                                      size = INPUTSPERGATE[gate],
                                                      replace = False))
                else:
                    arguments = list(np.random.choice(validInputs,
                                                      size = INPUTSPERGATE[gate],
                                                      replace = False))
                self.operations.append(tuple([gate] + arguments))
            self.name = "%d inputs ; "%self.numberOfInputs + \
                        " ; ".join("%s(%s)"%(o[0],",".join(map(str,o[1:])))
                                   for o in self.operations )

        xs = list(itertools.product(*[ [False,True] for _ in range(numberOfInputs) ]))
        
        ys = [ self.evaluate(x) for x in xs ]
        self.examples = zip(xs,ys)

        # the signature is invariant to the construction of the circuit and only depends on its semantics
        self.signature = tuple(ys)

    def __eq__(self,o): return self.name == o.name
    def __ne__(self,o): return not (self == o)

    def task(self):
        request = arrow(*[tbool for _ in range(self.numberOfInputs + 1) ])
        features = Circuit.extractFeatures(list(self.signature))
        return RegressionTask(self.name, request, self.examples, features = features, cache = True)
    @staticmethod
    def extractFeatures(ys):
        features = [ float(int(y)) for y in ys ]
        maximumFeatures = 2**MAXIMUMINPUTS
        return features + [-1.]*(maximumFeatures - len(features))

    def evaluate(self,x):
        x = list(reversed(x))
        outputs = []
        for z in self.operations:
            f = GATEFUN[z[0]]
            arguments = [ (outputs + x)[a] for a in z[1:] ]
            outputs.append(f(*arguments))
        return outputs[-1]

    def isConnected(self):
        def used(o):
            arguments = { j for j in o[1:] if j >= 0 }
            return arguments | { k for j in arguments for k in used(self.operations[j]) }
        
        if self.operations == []: return False
        usedIndices = used(self.operations[-1])
        return len(usedIndices) == len(self.operations) - 1


class FeatureExtractor(HandCodedFeatureExtractor):
    def _featuresOfProgram(program, t):
        numberOfInputs = len(t.functionArguments())
        xs = list(itertools.product(*[ [False,True] for _ in range(numberOfInputs) ]))
        f = program.evaluate([])
        ys = [ program.runWithArguments(x) for x in xs ]
        return Circuit.extractFeatures(ys)

class DeepFeatureExtractor(MLPFeatureExtractor):
    def __init__(self, tasks):
        super(DeepFeatureExtractor, self).__init__(tasks, H = 16)
    def _featuresOfProgram(self, program, tp):
        numberOfInputs = len(tp.functionArguments())
        xs = list(itertools.product(*[ [False,True] for _ in range(numberOfInputs) ]))
        ys = [ program.runWithArguments(x) for x in xs ]
        return Circuit.extractFeatures(ys)
        
                
if __name__ == "__main__":
    circuits = []
    import random
    random.seed(0)
    while len(circuits) < NUMBEROFTASKS:
        inputs = sampleDistribution(inputDistribution)
        gates = sampleDistribution(gateDistribution)
        newTask = Circuit(numberOfInputs = inputs,
                          numberOfGates = gates)
        if newTask not in circuits:
            circuits.append(newTask)
    eprint("Sampled %d circuits with %d unique functions"%(len(circuits),
                                                       len({t.signature for t in circuits })))
    tasks = [t.task() for t in circuits ]

    baseGrammar = Grammar.uniform(primitives)
    explorationCompression(baseGrammar, tasks,
                           outputPrefix = "experimentOutputs/circuit",
                           **commandlineArguments(frontierSize = 1000,
                                                  iterations = 10,
                                                  aic = 1.,
                                                  structurePenalty = 0.1,
                                                  CPUs = numberOfCPUs(),
                                                  featureExtractor = DeepFeatureExtractor,
                                                  topK = 2,
                                                  maximumFrontier = 100,
                                                  a = 2,
                                                  activation = "relu",
                                                  pseudoCounts = 5.))
    
    
