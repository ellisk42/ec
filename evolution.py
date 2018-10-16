from program import *
from grammar import *


from arithmeticPrimitives import *
from listPrimitives import *

from recognition import *

import torch.nn.functional as F


class EvolutionGuide(RecognitionModel):
    def __init__(self, featureExtractor, grammar, hidden=[64], activation="relu",
                 cuda=False, contextual=False):
        super(EvolutionGuide, self).__init__(featureExtractor, grammar,
                                             hidden=hidden, activation=activation,
                                             cuda=cuda, contextual=contextual)

        # value and policy
        self.value = nn.Linear(self.outputDimensionality, 1)
        self.policy = ContextualGrammarNetwork(self.outputDimensionality, grammar)

        if cuda: self.cuda()

    def mutationGrammar(self, goal, current):
        return self.policy(self._MLP(self.featureExtractor.featuresOfTask(goal, current)))
    def getFitness(self, goal, current):
        return self.value(self._MLP(self.featureExtractor.featuresOfTask(goal, current)))
    def mutationAndFitness(self, goal, current):
        features = self._MLP(self.featureExtractor.featuresOfTask(goal, current))
        return self.policy(features), self.value(features)
    

bootstrapTarget()
g = Grammar.uniform([Program.parse(p)
                     for p in ["+","-","0","1",
                               "fold","empty","cons"] ])


def children(g, request, _=None,
             ancestor=None, timeout=None):
    message = {"DSL": g.json(),
               "request": request.json(),
               "extras": [[]],
               "timeout": float(timeout)
    }
    if ancestor is not None: message["ancestor"] = str(ancestor)

    response = jsonBinaryInvoke("./evolution", message)
    children = []
    for e in response:
        mutation = Program.parse(e['programs'][0])
        if ancestor is None: child = mutation
        else: child = Application(mutation,ancestor)
        children.append(child)
    return children

def fitness(p):
    try:
        l = p.runWithArguments([])
    except: return -10
    reference = [-1,2,1,0]*2
    if len(l) < len(reference):
        l = l + [None]*(len(reference) - len(l))
    elif len(l) > len(reference):
        return -100
    for f,(x,y) in enumerate(zip(l,reference)):
        if x != y:
            if x is None:
                return f
            return -10
    return 100

    
population = []
timeout=20
best = 2
request = tlist(tint)
for generation in range(3):
    eprint(" ==  ==  ==  == ")
    eprint("Starting generation",generation)
    eprint("Current members of population:")
    for p in population:
        eprint(p, "\t", fitness(p))
    eprint(" ==  ==  ==  == ")
    eprint()

    if generation == 0:
        newPopulation = children(g, request, timeout=timeout)
    else:
        newPopulation = []
        for ancestor in population:
            newPopulation.extend(children(g, request, timeout=timeout,
                                          ancestor=ancestor))

    
    newPopulation.sort(key=fitness, reverse=True)
    population = newPopulation[:best]
    
