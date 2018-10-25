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

    def batchedForward(self, goal, currents):
        features = self._MLP(self.featureExtractor.featuresOfTasks([goal]*len(currents), currents))
        B = features.shape[0]
        v = self.value(features)
        return [self.policy(features[b]) for b in range(B) ], [v[b] for b in range(B) ]

    def graphForward(self, root):
        """Returns a dictionary of {node: (policy, value)}, for each node in the graph"""
        children = root.reachable()
        children = list(children)
        # Make sure that everything has a task associated with it
        for c in children:
            if c.current is None and c.program is not None:
                c.current = self.featureExtractor.taskOfProgram(c.program, c.goal.request,
                                                                lenient=True)
                assert c.current is not None

        goal = root.goal
        policies, values = self.batchedForward(goal, [c.current for c in children])
        return {c: (p,v)
                for c,p,v in zip(children, policies, values) }

    def batchedLoss(self, root):
        pv = self.graphForward(root)

        distance = {} # map from node in graph to distance
        def _distance(ev):
            if ev in distance: return distance[ev]
            if ev.isGoal:
                d = 0.
            else:
                alternatives = []
                mg = pv[ev][0]
                for edge in ev.descendents:
                    edgeCost = -edge.likelihoodSummary(self.generativeModel).logLikelihood(mg).view(-1)
                    alternatives.append(edgeCost + _distance(edge.child))
                d = torch.stack(alternatives,1).view(-1)
                d = d.squeeze(0).min(0)[0]
            distance[ev] = d
            return d
        pl = _distance(root)
        vl = sum( (distance[ev] - pv[ev][1])**2
                  for ev in root.reachable())
        return pl,vl

                
            
        
        
                

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

    # def policyLoss(self, ev, table=None):
    #     if ev.isGoal: return 0.
    #     if ev.current is None and ev.program is not None:
    #         ev.current = self.featureExtractor.taskOfProgram(ev.program, ev.goal.request)

    #     if table is None: table = {}

    #     alternatives = []
    #     mg = self.mutationGrammar(ev.goal, ev.current)
    #     request = ev.goal.request if ev.program is None else arrow(ev.goal.request, ev.goal.request)
    #     for mutation, child in ev.descendents:
    #         l = self.policyLoss(child)
    #         likelihoodSummary = self.grammar.closedLikelihoodSummary(request, mutation)
    #         l -= likelihoodSummary.logLikelihood(mg)
    #         alternatives.append(l)
            
    #     l = torch.stack(alternatives,1).squeeze(0)
    #     l = l.min(0)[0]
    #     l = l.unsqueeze(0)
    #     table[ev] = l
    #     return l

    # def valueLoss(self, ev):
    #     distanceTable = {}
    #     l = [0.]
    #     def distance(ev):
    #         if ev in distanceTable: return distanceTable[ev]
    #         request = ev.goal.request if ev.program is None else arrow(ev.goal.request, ev.goal.request)
            
    #         if ev.isGoal:
    #             d = 0.
    #         else:
    #             d = POSITIVEINFINITY
    #             mg = self.mutationGrammar(ev.goal, ev.current).untorch()
    #             for mutation, child in ev.descendents:
    #                 edgeCost = -self.grammar.closedLikelihoodSummary(request, mutation).logLikelihood(mg)
    #                 d = min(edgeCost + distance(child), d)
    #         distanceTable[ev] = d
    #         prediction = -self.getFitness(ev.goal, ev.current)
    #         l[0] += (d - prediction)**2
    #         return d

    #     distance(ev)
    #     return l[0]
    
    def train(self, graphs, lr=0.001):
        policy_optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-3, amsgrad=True)
        value_optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-3, amsgrad=True)

        losses = []
        while True:
            for ev in graphs:
                self.zero_grad()
                pl, vl = self.batchedLoss(ev)
                pl.backward()
                policy_optimizer.step()

                self.zero_grad()
                pl, vl = self.batchedLoss(ev)
                vl.backward()
                value_optimizer.step()
                # self.zero_grad()
                # vl = self.valueLoss(ev)
                # vl.backward()
                # value_optimizer.step()7
                                
                losses.append((pl.data.tolist(),
                               vl.data.tolist()[0]))
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

class EV:
    """evolution vertex: a vertex in the graph describing all evolutionary trajectories to a solution"""
    def __init__(self, goal, program):
        self.program = program
        self.goal = goal
        # outgoing edges
        self.descendents = []

        # current: task option
        # where we are currently in the search space
        self.current = None

        self.isGoal = False

    def __eq__(self,o):
        if self.program is None: return o.program is None
        if o.program is None: return False
        return self.program == o.program
    def __ne__(self,o): return not (self == o)
    def __hash__(self): return hash(self.program)

    def reachable(self, visited=None):
        if visited is None: visited = set()
        if self in visited: return visited
        visited.add(self)
        for d in self.descendents: d.child.reachable(visited)
        return visited

    class Edge:
        """evolutionary edge"""
        def __init__(self, ancestor, mutation, child, request):
            self.ancestor = ancestor
            self.mutation = mutation
            self.child = child
            self.request = request
            self._likelihoodSummary = None

        def likelihoodSummary(self, g):
            if self._likelihoodSummary is None:
                self._likelihoodSummary = g.closedLikelihoodSummary(self.request, self.mutation)
            return self._likelihoodSummary
        
def evolutionaryTrajectories(task, seed):
    request = task.request

    # map from program to EV
    # Initially we just have no program
    table = {None: EV(task, None)}

    def getVertex(p):
        if p in table: return table[p]
        v = EV(task,p)
        # Single step mutation that just gets us here in one shot
        table[None].descendents.append(EV.Edge(ancestor=None,
                                               mutation=p,
                                               child=v,
                                               request=request))
        table[p] = v
        for m,a in possibleAncestors(request,p):
            av = getVertex(a)
            av.descendents.append(EV.Edge(ancestor=av,
                                          mutation=m,
                                          child=v,
                                          request=arrow(request,request)))
        return v

    v = getVertex(seed)
    v.isGoal = True

    from graphviz import Digraph

    g = Digraph()

    for p,v in table.items():
        g.node(str(p))
    for p,v in table.items():
        for edge in v.descendents:
            g.edge(str(p),
                   str(edge.child.program),
                   label=str(edge.mutation))
    #g.render("/tmp/whatever.pdf",view=True)
    return table[None]        
    

from towerPrimitives import *
from makeTowerTasks import *
g = Grammar.uniform(primitives)
t = makeSupervisedTasks()[0]
p = Program.parse("(lambda (1x3 (right 4 (1x3 (left 2 (3x1 $0))))))")
eprint(g.logLikelihood(t.request,p))
trajectories = [evolutionaryTrajectories(t,
                                         p)]
from tower import TowerCNN


rm = EvolutionGuide(TowerCNN([]),g,contextual=False)
rm.train(trajectories)
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
    
