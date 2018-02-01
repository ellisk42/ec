from ec import explorationCompression, commandlineArguments
from grammar import Grammar
from utilities import eprint, sampleDistribution
from circuitPrimitives import primitives
from task import RegressionTask
from type import arrow, tbool

import itertools
import random

inputDistribution = [#(1,1),
#                     (2,2),
#                     (4,3),
#                     (4,4),
    (4,5),
    (4,6),
    (4,7)]
MAXIMUMINPUTS = max(i for p,i in inputDistribution)
gateDistribution = [(1,1),
                    (2,2),
                    (3,3),
#                    (4,3),
                    #(4,4),
                    #(5,5),
                    #(6,5),
]
operationDistribution = [(1,'NOT'),
                         (2,'AND'),
                         (2,'OR')]

class Circuit(object):
    def __init__(self, _ = None, numberOfInputs = None, numberOfGates = None):
        assert numberOfInputs != None
        assert numberOfGates != None

        self.numberOfInputs = numberOfInputs

        self.operations = []
        while not self.isConnected():
            self.operations = []
            while len(self.operations) < numberOfGates:
                gate = sampleDistribution(operationDistribution)
                x1 = random.choice(range(-numberOfInputs, len(self.operations)))
                x2 = random.choice(range(-numberOfInputs, len(self.operations)))
                if gate != 'NOT':
                    self.operations.append((gate,x1,x2))
                else:
                    self.operations.append((gate,x1))
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
            o = z[0]
            v1 = (outputs + x)[z[1]]
            v2 = (outputs + x)[z[-1]]
            if o == 'AND':
                outputs.append(v1 and v2)
            elif o == 'OR':
                outputs.append(v1 or v2)
            elif o == 'NOT':
                outputs.append(not v1)
            else:
                assert False
        return outputs[-1]

    def isConnected(self):
        def used(o):
            arguments = { j for j in o[1:] if j >= 0 }
            return arguments | { k for j in arguments for k in used(self.operations[j]) }
        
        if self.operations == []: return False
        usedIndices = used(self.operations[-1])
        return len(usedIndices) == len(self.operations) - 1


def makeFeatureExtractor((averages, deviations)):
    def featureExtractor(program, t):
        numberOfInputs = len(t.functionArguments())
        xs = list(itertools.product(*[ [False,True] for _ in range(numberOfInputs) ]))
        f = program.evaluate([])
        ys = []
        for x in xs:
            y = f
            for x_ in x:
                y = y(x_)
            ys.append(y)
        return RegressionTask.standardizeFeatures(averages, deviations, Circuit.extractFeatures(ys))
    return featureExtractor
                
if __name__ == "__main__":
    tasks = []
    while len(tasks) < 1000:
        inputs = sampleDistribution(inputDistribution)
        gates = sampleDistribution(gateDistribution)
        newTask = Circuit(numberOfInputs = inputs,
                          numberOfGates = gates)
        if newTask not in tasks:
            tasks.append(newTask)
    eprint("Sampled %d tasks with %d unique functions"%(len(tasks),
                                                       len({t.signature for t in tasks })))
    tasks = [t.task() for t in tasks ]

    statistics = RegressionTask.standardizeTasks(tasks)
    featureExtractor = makeFeatureExtractor(statistics)

    baseGrammar = Grammar.uniform(primitives)
    explorationCompression(baseGrammar, tasks,
                           outputPrefix = "experimentOutputs/circuit",
                           **commandlineArguments(frontierSize = 500,
                                                  iterations = 10,
                                                  aic = 1.,
                                                  structurePenalty = 0.1,
                                                  featureExtractor = featureExtractor,
                                                  topK = 2,
                                                  maximumFrontier = 100,
                                                  a = 2,
                                                  activation = "tanh",
                                                  pseudoCounts = 5.))
    
    
