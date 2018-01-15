from ec import *

import itertools
import random

class Circuit(RegressionTask):
    def __init__(self, _ = None, numberOfInputs = None, numberOfGates = None):
        assert numberOfInputs != None
        assert numberOfGates != None

        self.numberOfInputs = numberOfInputs

        self.operations = []
        while not self.isConnected():
            self.operations = []
            while len(self.operations) < numberOfGates:
                gate = random.choice(['OR','AND','NOT'])
                x1 = random.choice(range(-numberOfInputs, len(self.operations)))
                x2 = random.choice(range(-numberOfInputs, len(self.operations)))
                if gate != 'NOT':
                    self.operations.append((gate,x1,x2))
                else:
                    self.operations.append((gate,x1))
            self.name = " ; ".join("%s(%s)"%(o[0],",".join(map(str,o[1:])))
                                   for o in self.operations )

        xs = list(itertools.product(*[ [False,True] for _ in range(numberOfInputs) ]))
        
        ys = [ self.evaluate(x) for x in xs ]
        examples = zip(xs,ys)

        # the signature is invariant to the construction of the circuit and only depends on its semantics
        self.signature = tuple(ys)

        request = arrow(*[tbool for _ in range(numberOfInputs + 1) ])        
        super(Circuit, self).__init__(self.name, request, examples, cache = True)

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
        if len(usedIndices) == len(self.operations) - 1:
            print self.name
            print usedIndices
            print 
        return len(usedIndices) == len(self.operations) - 1
                
primitives = [Primitive("nand",arrow(tbool,tbool,tbool),
                        lambda x: lambda y: not (x and y))]

if __name__ == "__main__":
    tasks = []
    while len(tasks) < 1000:
        inputs = random.choice([1,2,3])
        gates = random.choice([1,2,3,4,5])
        tasks.append(Circuit(numberOfInputs = inputs,
                             numberOfGates = gates))
    print "Sampled %d tasks with %d unique functions"%(len(tasks),
                                                       len({t.signature for t in tasks }))
    explorationCompression(primitives, tasks,
                           **commandlineArguments(frontierSize = 10**3,
                                                  iterations = 5,
                                                  pseudoCounts = 5.))
    
    
