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

    def children(self, goal, _=None,
                 ancestor=None, timeout=None):
        g = self.mutationGrammar(goal, ancestor).untorch()
        message = {"DSL": g.json(),
                   "request": goal.request.json(),
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

    def search(self, goal, _=None,
               populationSize=None, timeout=None, generations=None):
        assert populationSize is not None
        assert timeout is not None
        assert generations is not None

        # Map from parent to fitness
        population = {None: 1.}
        everyChild = set()

        for _ in range(generations):
            z = sum(population.values())
            children = []
            for ancestor, fitness in population.items():
                children.extend(self.children(goal, ancestor,
                                              timeout=timeout*fitness/z))
            population = {child: self.getFitness(goal, self.taskOfProgram(child)).view(-1).data[0]
                          for child in set(children) }
            for child in population: everyChild.add(child)

        return everyChild
            
                
def possibleAncestors(request, program):
    from itertools import permutations

    program = program.clone()
    context = MutableContext()
    program.annotateTypes(context, [])
    eprint(program.annotatedType)
    
    desiredNumberOfArguments = len(request.functionArguments())
    def curse(d, p):
        # Returns a set of (mutation, ancestor)
        parses = set()

        # Could this be the ancestor?
        freeVariableTypes = {}
        tp = p.annotatedType
        if len(freeVariableTypes) + len(tp.functionArguments()) == desiredNumberOfArguments:
            for fv in permutations( (fi, ft.apply(context))
                                    for fi, ft in freeVariableTypes.items()):
                t = tp
                for _,fvt in reversed(fv): t = arrow(fvt,t)
                if canUnify(t, request):
                    parses.add((Index(d), p))

        if p.isIndex:
            parses.add((Index(p.i + 1),None))
        if p.isPrimitive:
            parses.add((p,None))
        if p.isApplication:
            f = curse(d, p.f)
            x = curse(d, p.x)
            for fp,fa in f:
                for xp,xa in x:
                    if fa is not None and \
                       xa is not None and \
                       fa != xa:
                        continue
                    a = fa or xa
                    parses.add((Application(fp,xp), a))
        if p.isAbstraction:
            for b,a in curse(d + 1, p.body):
                parses.add((Abstraction(b), a))
        return parses

    return curse(0, program)


bootstrapTarget()
g = Grammar.uniform([Program.parse(p)
                     for p in ["+","-","0","1","car",
                               "fold","empty","cons"] ])

eprint(possibleAncestors(arrow(tlist(tint),tlist(tint)),
                         Program.parse("(lambda (fold $0 (cons (car $0) empty) (lambda (lambda (cons $1 $0)))))")))
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
    
