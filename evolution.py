from program import *
from grammar import *


from arithmeticPrimitives import *
from utilities import *
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
        if contextual:
            self.policy = ContextualGrammarNetwork(self.outputDimensionality, grammar)
        else:
            self.policy = GrammarNetwork(self.outputDimensionality, grammar)

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

    def policyLoss(self, ev, table=None):
        if ev.isGoal: return 0.
        if ev.current is None and ev.program is not None:
            ev.current = self.featureExtractor.taskOfProgram(ev.program, ev.goal.request)

        if table is None: table = {}

        alternatives = []
        mg = self.mutationGrammar(ev.goal, ev.current)
        request = ev.goal.request if ev.program is None else arrow(ev.goal.request, ev.goal.request)
        for mutation, child in ev.descendents:
            l = self.policyLoss(child)
            likelihoodSummary = self.grammar.closedLikelihoodSummary(request, mutation)
            l -= likelihoodSummary.logLikelihood(mg)
            alternatives.append(l)
            
        l = torch.stack(alternatives,1).squeeze(0)
        l = l.min(0)[0]
        l = l.unsqueeze(0)
        table[ev] = l
        return l

    def valueLoss(self, ev):
        distanceTable = {}
        def distance(ev):
            if ev in distanceTable: return distanceTable[ev]
            request = ev.goal.request if ev.program is None else arrow(ev.goal.request, ev.goal.request)
            
            if ev.isGoal:
                d = 0.
            else:
                d = POSITIVEINFINITY
                mg = self.mutationGrammar(ev.goal, ev.current)
                for mutation, child in ev.descendents:
                    edgeCost = -self.grammar.closedLikelihoodSummary(request, mutation).logLikelihood(mg)
                    edgeCost = edgeCost.view(-1).data.tolist()[0]
                    d = min(edgeCost + distance(child), d)
            distanceTable[ev] = d
            return d

        
                
        if ev.current is None and ev.program is not None:
            ev.current = self.featureExtractor.taskOfProgram(ev.program, ev.goal.request)

        if table is None: table = {}

        alternatives = []
        mg = self.mutationGrammar(ev.goal, ev.current)
        request = ev.goal.request if ev.program is None else arrow(ev.goal.request, ev.goal.request)
        for mutation, child in ev.descendents:
            l = self.policyLoss(child)
            likelihoodSummary = self.grammar.closedLikelihoodSummary(request, mutation)
            l -= likelihoodSummary.logLikelihood(mg)
            alternatives.append(l)
            
        l = torch.stack(alternatives,1).squeeze(0)
        l = l.min(0)[0]
        l = l.unsqueeze(0)
        table[ev] = l
        return l

    def trainPolicy(self, graphs, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-3, amsgrad=True)

        losses = []
        while True:
            for ev in graphs:
                self.zero_grad()
                loss = self.policyLoss(ev)
                loss.backward()
                optimizer.step()
                losses.append(loss.data.tolist()[0])
                eprint(losses[-1])

        

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
    def annotateIndices(p):
        if p.isIndex:
            p.variableTypes = {p.i: p.annotatedType.applyMutable(context)}
        elif p.isPrimitive or p.isInvented:
            p.variableTypes = dict()
        elif p.isAbstraction:
            annotateIndices(p.body)
            p.variableTypes = {(i - 1): t
                               for i,t in p.body.variableTypes.items()
                               if i > 0}
        elif p.isApplication:
            annotateIndices(p.f)
            annotateIndices(p.x)
            p.variableTypes = {i: p.f.variableTypes.get(i, p.x.variableTypes.get(i, None))
                               for i in set(list(p.f.variableTypes.keys()) + list(p.x.variableTypes.keys()))}
        else: assert False

    annotateIndices(program)

    def renameAncestorVariables(d,a, mapping):
        if a.isIndex:
            if a.i - d >= 0:
                return Index(mapping[a.i - d])
            return a
        if a.isApplication:
            return Application(renameAncestorVariables(d,a.f,mapping),
                               renameAncestorVariables(d,a.x,mapping))
        if a.isAbstraction:
            return Abstraction(renameAncestorVariables(d + 1, a.body, mapping))
        if a.isPrimitive or a.isInvented:
            return a
        assert False
    
    desiredNumberOfArguments = len(request.functionArguments())
    def curse(d, p):
        # Returns a set of (mutation, ancestor)
        parses = set()

        # Could this be the ancestor?
        freeVariableTypes = p.variableTypes
        tp = p.annotatedType
        if not p.isIndex and \
           len(freeVariableTypes) + len(tp.functionArguments()) == desiredNumberOfArguments:
            for fv in permutations(freeVariableTypes.items()):
                t = tp
                for _,fvt in reversed(fv): t = arrow(fvt,t)
                if canUnify(t, request):
                    # Apply the ancestor
                    m = Index(d)
                    for fi,_ in fv: m = Application(m,Index(fi))
                    # rename variables inside of ancestor
                    mapping = {fi: fi_ for fi_,(fi,_) in enumerate(reversed(fv)) }
                    a = renameAncestorVariables(0, p, mapping)
                    for _ in fv: a = Abstraction(a)
                    a = EtaLongVisitor(request).execute(a)
                    parses.add((m, a))

        if p.isIndex or p.isPrimitive or p.isInvented:
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

    return {(EtaLongVisitor(arrow(request, request)).execute(Abstraction(m).clone()),
             a.clone())
            for m,a in curse(0, program)
            if a is not None and m != Index(0) and a != program}

class EV():
    """evolution vertex: a vertex in the graph describing all evolutionary trajectories to a solution"""
    def __init__(self, goal, program):
        self.program = program
        self.goal = goal
        # outgoing edges
        # descendents: [(mutation, EV)]
        self.descendents = set()

        # current: task option
        # where we are currently in the search space
        self.current = None

        self.isGoal = False

    def __eq__(self,o): return self.program == o.program
    def __ne__(self,o): return not (self == o)
    def __hash__(self): return hash(self.program)
        
def evolutionaryTrajectories(task, seed):
    request = task.request
    
    # map from program to EV
    # Initially we just have no program
    table = {None: EV(task, None)}

    def getVertex(p):
        if p in table: return table[p]
        v = EV(task,p)
        # Single step mutation that just gets us here in one shot
        table[None].descendents.add((p,v))
        table[p] = v
        for m,a in possibleAncestors(request,p):
            av = getVertex(a)
            av.descendents.add((m,v))
        return v

    v = getVertex(seed)
    v.isGoal = True

    from graphviz import Digraph

    g = Digraph()

    for p,v in table.items():
        g.node(str(p))
    for p,v in table.items():
        for m,d in v.descendents:
            g.edge(str(p), str(d.program),
                   label=str(m) if p is not None else None)
    g.render("/tmp/whatever.pdf",view=True)
    return table[None]        
    

from towerPrimitives import *
from makeTowerTasks import *
g = Grammar.uniform(primitives)
t = makeSupervisedTasks()[0]

trajectories = [evolutionaryTrajectories(t,
                         Program.parse("(lambda (1x3 (right 4 (1x3 (left 2 (3x1 $0))))))"))]
from tower import TowerCNN


rm = EvolutionGuide(TowerCNN([]),g,contextual=False)
rm.trainPolicy(trajectories)
        # eprint("Possible trajectory:")
        # for m in reversed(ts):
        #     eprint(m)
        # eprint()
    # for m,a in possibleAncestors(r,
    #                              a):
    #     eprint("ancestor",a)
    #     eprint("mutation",m)
    #     eprint()
assert False
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
    
