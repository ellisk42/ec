from program import *
from grammar import *


from arithmeticPrimitives import *
from utilities import *
from listPrimitives import *

from recognition import *

import torch.nn.functional as F

def extract_scaler(v):
    v = v.view(-1)
    assert v.shape == (1,)
    v = v.data.tolist()[0]
    return v

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
                    edgeCost = -edge.likelihoodSummary(self.grammar).logLikelihood(mg).view(-1)
                    alternatives.append(edgeCost + _distance(edge.child))
                if False:
                    d = torch.stack(alternatives,1).view(-1)
                    d = d.squeeze(0).min(0)[0]
                else:
                    d = -torchSoftMax([-a for a in alternatives ])
            distance[ev] = d
            return d
        pl = _distance(root)
        vl = sum( (distance[ev] - pv[ev][1])**2
                  for ev in root.reachable())
        vl = vl/len(distance) # MSE
        return pl,vl

    def visualize(self, root):
        pv = self.graphForward(root)

        actualDistance = {}
        predictedDistance = {}
        edgeCost = {}
        
        def analyze(ev):
            if ev in predictedDistance: return
            predictedDistance[ev] = extract_scaler(pv[ev][1])
            
            if ev.isGoal:
                actualDistance[ev] = 0.
            else:
                alternatives = []
                mg = pv[ev][0]
                for edge in ev.descendents:
                    ec = -edge.likelihoodSummary(self.generativeModel).logLikelihood(mg).view(-1)
                    ec = extract_scaler(ec)
                    edgeCost[edge] = ec
                    analyze(edge.child)
                    alternatives.append(ec + actualDistance[edge.child])
                
                actualDistance[ev] = min(alternatives)
            
        analyze(root)

        from graphviz import Digraph
        g = Digraph()

        def name(ev):
            return "%s\nV*=%f\nV=%f"%(ev.program,
                                      actualDistance[ev],
                                      predictedDistance[ev])

        for ev in actualDistance:
            g.node(name(ev))
        for ev in actualDistance:
            if len(ev.descendents) == 0: continue
            
            bestEdge = min(ev.descendents, key=lambda e: edgeCost[e])
            for edge in ev.descendents:
                g.edge(name(ev),
                       name(edge.child),
                       label="%s\n%f"%(edge.mutation,
                                       edgeCost[edge]),
                       color="red" if edge == bestEdge else "black")
        g.render("/tmp/evolutionGraph.pdf",view=True)
        

                
            
        
        
                

    def children(self, _=None,
                 ancestor=None, timeout=None):
        """
        ancestor: EV.
        returns: list of programs built from ancestor
        """
        g = self.policy(self._MLP(self.featureExtractor.featuresOfTask(ancestor.goal, ancestor.current))).untorch()
        message = {"DSL": g.json(),
                   "request": ancestor.goal.request.json(),
                   "extras": [[]],
                   "timeout": float(timeout)
        }
        if ancestor.program is not None: message["ancestor"] = str(ancestor.program)
        if self.featureExtractor.__class__.special:
            message["special"] = self.featureExtractor.__class__.special

        response = jsonBinaryInvoke("./evolution", message)
        children = []
        for e in response:
            mutation = Program.parse(e['programs'][0])
            if ancestor.program is None: child = mutation
            else: child = Application(mutation,ancestor.program)
            children.append(child)
        return children

    def train(self, graphs, _=None,
              lr=0.001, timeout=None, steps=None):
        policy_optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-3, amsgrad=True)
        value_optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-3, amsgrad=True)

        startTime = time.time()
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

                if steps and len(losses) > steps or timeout and time.time() - startTime > timeout:
                    return 

        
    def search(self, goal, _=None,
               populationSize=None, timeout=None, generations=None,
               fitnessBatch=100):
        assert populationSize is not None
        assert timeout is not None
        assert generations is not None

        # Map from parent to fitness
        population = {EV(goal, None): 1.}
        everyChild = set()

        for _ in range(generations):
            z = sum(population.values())
            children = []
            for ancestor, fitness in population.items():
                numberOfChildren = 0
                for child in self.children(ancestor=ancestor,
                                           timeout=timeout*fitness/z):
                    ev = EV(goal, child)
                    ev.current = self.featureExtractor.taskOfProgram(child, goal.request)
                    if ev.current is not None:
                        children.append(ev)
                        numberOfChildren += 1
                eprint("Ancestor",ancestor.program,
                       "produced",numberOfChildren,"children.")
                
            children = list(set(children))
            eprint("All of the ancestors collectively produced",len(children),
                   "new children.")

            # Keep only populationSize children

            bestChildren = PQ()
            childIndex = 0
            while childIndex < len(children):
                childBatch = children[childIndex:childIndex + fitnessBatch]
                childTasks = [ child.current for child in childBatch ]
                childFeatures = self.featureExtractor.featuresOfTasks([goal]*len(childTasks), childTasks)
                ds = self.value(self._MLP(childFeatures))
                ds = ds.data.view(-1).tolist()
                for child, d, batchIndex in zip(childBatch, ds, range(len(childBatch))):
                    f = -d
                    eprint("Child",child.program,"has fitness",f)
                    bestChildren.push(-f, (f, childIndex + batchIndex, child))
                    if len(bestChildren) > populationSize:
                        _1, worstChildIndex, _2 = bestChildren.popMaximum()
                        children[worstChildIndex] = None # garbage collect
                childIndex += fitnessBatch
                    
            population = {}
            for f,_,child in bestChildren:
                everyChild.add(child)
                population[child] = f

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

    def removeLongPaths(self, maxPath):
        assert self.program is None

        shortestPath = {}
        def curse(v):
            if v in shortestPath:
                assert shortestPath[v] <= maxPath
                return shortestPath[v]
            if v.isGoal:
                shortestPath[v] = 0
                return 0

            for d in v.descendents: curse(d.child)
            v.descendents = [d for d in v.descendents
                             if shortestPath[d.child] + 1 <= maxPath]
            shortestPath[v] = min( shortestPath[c.child] for c in v.descendents ) + 1
            assert shortestPath[v] <= maxPath
            return shortestPath[v]
        curse(self)
            

    class Edge:
        """evolutionary edge"""
        def __init__(self, ancestor, mutation, child, request):
            self.ancestor = ancestor
            self.mutation = mutation
            self.child = child
            self.request = request
            self._likelihoodSummary = None

        def __eq__(self, o):
            return (self.ancestor, self.mutation, self.child) == \
                (o.ancestor, o.mutation, o.child)
        def __ne__(self, o): return not (self == o)
        def __hash__(self): return hash((self.ancestor, self.mutation, self.child))

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

    return table[None]        
    

from towerPrimitives import *
from makeTowerTasks import *
g = Grammar.uniform(primitives)
t = makeSupervisedTasks()[0]
p = Program.parse("(lambda (1x3 (right 4 (1x3 (left 2 (3x1 $0))))))")
eprint(g.logLikelihood(t.request,p))
trajectory = evolutionaryTrajectories(t,p)
trajectory.removeLongPaths(2)
trajectories = [trajectory]
from tower import TowerCNN


rm = EvolutionGuide(TowerCNN([]),g,contextual=True)
rm.train(trajectories, timeout=30)
rm.visualize(trajectory)
rm.search(trajectory.goal,
          populationSize=10,
          timeout=3,
          generations=3)

