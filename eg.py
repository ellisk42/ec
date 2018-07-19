from grammar import *
from program import *
from utilities import *
from frontier import *

from frozendict import frozendict

def instantiate(context, environment, tp):
    bindings = {}
    context, tp = tp.instantiate(context, bindings)
    newEnvironment = {}
    for i,ti in environment.items():
        context,newEnvironment[i] = ti.instantiate(context, bindings)
    return context, newEnvironment, tp

def unify(*environmentsAndTypes):
    k = Context.EMPTY
    e = {}
    k,t = k.makeVariable()
    for e_,t_ in environmentsAndTypes:
        k, e_, t_ = instantiate(k, e_, t_)
        k = k.unify(t,t_)
        for i,ti in e_.items():
            if i not in e: e[i] = ti
            else: k = k.unify(e[i], ti)
    return {i: ti.apply(k) for i,ti in e.items() }, t.apply(k)




class Class():
    nextName = 0
    def __init__(self):
        self.name = Class.nextName
        self.leader = None
        Class.nextName += 1
    def __hash__(self): return self.name
    def __str__(self): return f"E{self.name}"
    def __eq__(self,o): return self.name == o.name
    def __ne__(self,o): return not (self == o)

    def chase(self):
        if self.leader is None: return self
        k = self.leader
        while k.leader is not None:
            k = k.leader
        self.leader = k
        return k
    
class EquivalenceGraph():
    EPSILONCOST = 0.00001
    def __init__(self, typed=True):
        self.typed = typed
        
        # Map from equivalence class to a set of expressions
        self.classMembers = {}
        # Map from expression to a set of equivalence classes
        self.classes = {}
        # Map from equivalence class to the set of all expressions incident on that class
        self.incident = {}

        # if an expression belongs to more than one class than it is in this table
        self.numberOfClasses = {}

        # external sets referring to equivalence classes
        self.externalUsers = {}

        # map from a equivalents class to its principal type
        # principal types are (environment, return type)
        self.typeOfClass = {}
        # ditto
        self.typeOfExpression = {}

    def inferClass(self,k):
        return self.typeOfClass[k]
    def inferExpression(self,l):
        if not self.typed: return (None,None)
        
        if l in self.typeOfExpression: return self.typeOfExpression[l]

        if l.isIndex: T = ({l.i: t0}, t0)
        elif l.isPrimitive or l.isInvented: T = ({}, l.tp)
        elif l.isAbstraction:
            Tb = self.inferClass(l.body)
            assert Tb is not None
            be,bt = Tb
            k = Context.EMPTY
            k,be,bt = instantiate(k,be,bt)
            if 0 in be:
                argumentType = be[0]
            else:
                k,argumentType = k.makeVariable()

            T = ({i - 1: ti
                  for i,ti in be.items()
                  if i > 0},
                 arrow(argumentType, bt))
        elif l.isApplication:
            Tf = self.inferClass(l.f)
            Tx = self.inferClass(l.x)
            assert Tf is not None
            assert Tx is not None

            k = Context.EMPTY
            k,fe,ft = instantiate(k, *Tf)
            k,xe,xt = instantiate(k, *Tx)

            k,value = k.makeVariable()
            try:
                k = k.unify(ft,arrow(xt, value))

                environment = dict(fe)
                for n,nt in xe.items():
                    if n in environment:
                        k = k.unify(environment[n],nt)
                    else:
                        environment[n] = nt

                T = ({n: nt.apply(k)
                      for n,nt in environment.items() },
                     value.apply(k))
            except UnificationFailure: T = None
        else: assert False

        if T is None:
            self.typeOfExpression[l] = None
            return None
        
        _,e,t = instantiate(Context.EMPTY, *T)
        self.typeOfExpression[l] = (frozendict(e),t)
        return e,t

        

        


            
            
                
        

    def setOfClasses(self,ks):
        ks = {k.chase() for k in ks if k is not None}
        for k in ks:
            if k not in self.externalUsers: self.externalUsers[k] = []
            self.externalUsers[k].append(ks)
        return ks

    def makeClass(self, tp):
        k = Class()
        self.classMembers[k] = set()
        self.incident[k] = set()
        self.typeOfClass[k] = tp
        return k

    def addEdge(self,k,l):
        self.classMembers[k].add(l)
        self.classes[l].add(k)
        assert self.inferClass(k) == self.inferExpression(l)
        if len(self.classes[l]) > 1:
            self.numberOfClasses[l] = len(self.classes[l])
    def deleteEdge(self,k,l):
        self.classMembers[k].remove(l)
        self.classes[l].remove(k)
        if l in self.numberOfClasses:
            n = len(self.classes[l])
            if n > 1:
                self.numberOfClasses[l] = n
            else:
                del self.numberOfClasses[l]                
    def deleteClass(self,k):
        assert len(self.classMembers[k]) == 0
        del self.classMembers[k]
        assert len(self.incident[k]) == 0
        del self.incident[k]
        del self.typeOfClass[k]

    def deleteExpression(self,l):
        assert len(self.classes[l]) == 0
        if l.isApplication:
            self.incident[l.f].remove(l)
            self.incident[l.x].remove(l)
        elif l.isAbstraction:
            self.incident[l.body].remove(l)
        del self.classes[l]
        if self.typed: del self.typeOfExpression[l]

    def rename(self, old, new):
        for refersToOld in list(self.classes[old]):
            self.deleteEdge(refersToOld, old)
            self.addEdge(refersToOld, new)
        self.deleteExpression(old)

        

    def incorporateClass(self,l):
        if self.inferExpression(l) is None: return None
        if l not in self.classes: self.classes[l] = set()
        if len(self.classes[l]) == 0:
            k = self.makeClass(self.inferExpression(l))
            self.addEdge(k,l)
            return k
        return getOne(self.classes[l])
    def incorporateExpression(self,l):
        if l in self.classes: return l
        t = self.inferExpression(l)
        if t is None: return None
        self.classes[l] = set()
        if l.isApplication:
            self.incident[l.f].add(l)
            self.incident[l.x].add(l)
        elif l.isAbstraction:
            self.incident[l.body].add(l)
        return l

    def applyClass(self,f,x):
        f = f.chase()
        x = x.chase()
        l = Application(f,x)
        if self.inferExpression(l) is None: return None
        self.incident[f].add(l)
        self.incident[x].add(l)
        return self.incorporateClass(Application(f,x))
    def abstractClass(self,b):
        b = b.chase()
        l = Abstraction(b)
        if self.inferExpression(l) is None: return None
        self.incident[b].add(l)
        return self.incorporateClass(l)
    def abstractLeaf(self,l):
        assert l.isIndex or l.isPrimitive or l.isInvented
        return self.incorporateClass(l)
    def incorporate(self,e):
        if e.isApplication:
            return self.applyClass(self.incorporate(e.f),
                                   self.incorporate(e.x))
        if e.isAbstraction:
            return self.abstractClass(self.incorporate(e.body))
        return self.abstractLeaf(e)

    def makeEquivalent(self,k1,k2):
        k1 = k1.chase()
        k2 = k2.chase()
        if k1 == k2: return k1
        if self.typeOfClass[k1] != self.typeOfClass[k2]: assert False

        # k2 is going to be deleted
        k2.leader = k1
        k2members = list(self.classMembers[k2])
        for l in k2members:
            self.addEdge(k1,l)
            self.deleteEdge(k2,l)
        
        def update(l):
            if l.isApplication:
                assert l.f == k2 or l.x == k2
                f = k1 if l.f == k2 else l.f
                x = k1 if l.x == k2 else l.x
                return self.incorporateExpression(Application(f,x))
            elif l.isAbstraction:
                assert l.body == k2
                return self.incorporateExpression(Abstraction(k1))
            else: assert False

        oldExpressions = self.incident[k2]
        for old in list(oldExpressions):
            new = update(old)
            self.rename(old, new)

        self.deleteClass(k2)

        if k2 in self.externalUsers:
            for s in self.externalUsers[k2]:
                s.remove(k2)
                s.add(k1)
            del self.externalUsers[k2]

        self.merge()

        return k1

    def merge(self):
        while len(self.numberOfClasses) > 0:
            l = next(iter(self.numberOfClasses))
            ks = list(self.classes[l])
            assert len(ks) == self.numberOfClasses[l]
            for j in range(1,len(ks)):
                self.makeEquivalent(ks[0],ks[j])

    def minimumCosts(self, given, oldTable=None):
        if oldTable is None:
            basicClasses = {k
                            for k,children in self.classMembers.items()
                            if k in given or any( l.isIndex or l.isPrimitive or l.isInvented
                                                  for l in children )}
            table = {k: 1 if k in basicClasses else POSITIVEINFINITY for k in self.classMembers }
        else:
            basicClasses = given
            table = {g: 1 for g in given}

        def indexTable(k):
            if k in table: return table[k]
            return oldTable[k]

        def expressionCost(l):
            if l.isApplication: return indexTable(l.f) + indexTable(l.x)
            if l.isAbstraction: return indexTable(l.body)
            if l.isPrimitive or l.isInvented: return 1
            if l.isIndex: return 1
            assert False
        def relax(e):
            old = indexTable(e)
            new = old
            for l in self.classMembers[e]:
                new = min(expressionCost(l),new)
            return new, new < old

        q = {getOne(self.classes[i])
             for b in basicClasses
             for i in self.incident[b] }
        numberOfRelaxations = 0
        while True:
            if len(q) == 0:
                for k in table:
                    _,change = relax(k)
                    assert not changed
                return table
            
            n = getOne(q)
            q.remove(n)
            new, changed = relax(n)
            numberOfRelaxations += 1
            if changed:
                table[n] = new
                q.update(getOne(self.classes[i])
                         for i in self.incident[n])

    def betaLongCost(self, given, oldTable=None):
        # table[k] = (functionCost, argumentCost)
        if oldTable is None:
            basicClasses = {k
                            for k,children in self.classMembers.items()
                            if k in given or any( l.isIndex or l.isPrimitive or l.isInvented
                                                  for l in children )}
            table = {k: (1,1) if k in basicClasses else (POSITIVEINFINITY,POSITIVEINFINITY)
                     for k in self.classMembers }
        else:
            basicClasses = given
            table = {g: (1,1) for g in given}

        def argumentCost(k):
            if k in table: return table[k][1]
            return oldTable[k][1]
        def functionCost(k):
            if k in table: return table[k][0]
            return oldTable[k][0]

        def expressionCost(l):
            if l.isApplication: return functionCost(l.f) + argumentCost(l.x) + EquivalenceGraph.EPSILONCOST
            if l.isAbstraction: return argumentCost(l.body) + EquivalenceGraph.EPSILONCOST
            if l.isPrimitive or l.isInvented: return 1
            if l.isIndex: return 1
            assert False
        def relax(e):
            ac, fc = argumentCost(e), functionCost(e)
            new_ac, new_fc = ac, fc
            for l in self.classMembers[e]:
                lc = expressionCost(l)
                new_ac = min(lc,new_ac)
                if not l.isAbstraction:
                    new_fc = min(lc,new_fc)
            return (new_fc,new_ac), new_fc < fc, new_ac < ac

        q = {getOne(self.classes[i])
             for b in basicClasses
             for i in self.incident[b] }
        numberOfRelaxations = 0
        while True:
            if len(q) == 0:
                if False:
                    for k in table:
                        _,changeF, changeA = relax(k)
                        assert not changedF
                        assert not changedA
                #print(f"Relaxed {numberOfRelaxations}")
                return table
            
            n = getOne(q)
            q.remove(n)
            new, changedF, changedA = relax(n)
            numberOfRelaxations += 1
            if changedF or changedA:
                table[n] = new
                for i in self.incident[n]:
                    if (changedF and i.isApplication and i.f == n) or \
                       (changedA and ((not i.isApplication) or i.x == n)):
                        q.add(getOne(self.classes[i]))
        
        
    def extract(self,k,table=None):
        if table is None: table = self.minimumCosts([])
        def expressionCost(l):
            if l.isApplication: return table[l.f] + table[l.x] + EquivalenceGraph.EPSILONCOST
            if l.isAbstraction: return table[l.body] + EquivalenceGraph.EPSILONCOST
            if l.isPrimitive or l.isInvented: return 1
            if l.isIndex: return 1
            assert False
        def visitClass(k):
            return visitExpression(min(self.classMembers[k],
                                       key=expressionCost))
        def visitExpression(e):
            if e.isApplication:
                return Application(visitClass(e.f),
                                   visitClass(e.x))
            if e.isAbstraction:
                return Abstraction(visitClass(e.body))
            return e
        return visitClass(k)

    def extractBetaLong(self,k,table=None, given=[]):
        if table is None: table = self.betaLongCost(given)
        def functionCost(k): return table[k][0]
        def argumentCost(k): return table[k][1]

        def expressionCost(l):
            if l.isApplication: return functionCost(l.f) + argumentCost(l.x) + EquivalenceGraph.EPSILONCOST
            if l.isAbstraction: return argumentCost(l.body) + EquivalenceGraph.EPSILONCOST
            if l.isPrimitive or l.isInvented: return 1
            if l.isIndex: return 1
            assert False
        def visitClass(k):
            # eprint(f"Visiting class {k} w/ fc={functionCost(k)} & ac={argumentCost(k)}")
            # eprint(f"\t{self.classMembers[k]}")
            return visitExpression(min(self.classMembers[k],
                                       key=expressionCost))
        def visitExpression(e):
            if e.isApplication:
                return Application(visitClass(e.f),
                                   visitClass(e.x))
            if e.isAbstraction:
                return Abstraction(visitClass(e.body))
            return e
        return visitClass(k)

    def reachable(self,heads):
        r = set()
        def visit(k):
            if isinstance(k,Program):
                if k.isApplication:
                    visit(k.f)
                    visit(k.x)
                elif k.isAbstraction:
                    visit(k.body)
                return 
            if k in r: return
            r.add(k)
            for e in self.classMembers[k]:
                visit(e)

        for h in heads:
            visit(h)
        return r

    def solveInvention(self,heads):
        candidates = [self.reachable([h])
                      for h in heads ]

    def bestInvention(self,heads):
        referenceTable = self.betaLongCost([])
        def score(k):
            t = self.betaLongCost([k], oldTable=referenceTable)
            s = sum(min(t.get(h,referenceTable[h])) for h in heads )
            return s
        candidates = [self.reachable([h])
                      for h in heads ]
        from collections import Counter
        candidates = Counter(k for ks in candidates for k in ks)
        candidates = list({k for k,f in candidates.items() if f >= 2 })
        print(f"{len(candidates)} candidates")
        candidates = [(score(k),k) for k in candidates ]
        best = min(candidates,key = lambda s: s[0])[1]
        return self.extract(best)

    def bestInventions(self,heads,K):
        """heads: list of list of (indices to) programs; get these using the incorporate method.
        K: number of inventions to return
        returns: K expressions, _not_ wrapped in Invention"""
        referenceTable = self.betaLongCost([])
        def score(k):
            t = self.betaLongCost([k], oldTable=referenceTable)
            s = sum(min(min(t.get(h,referenceTable[h])) for h in hs ) for hs in heads )
            return s
        candidates = [self.reachable(hs)
                      for hs in heads ]
        from collections import Counter
        candidates = Counter(k for ks in candidates for k in ks)
        candidates = list({k for k,f in candidates.items() if f >= 2 })
        print(f"{len(candidates)} candidates")
        if len(candidates) < 1: return []
        
        candidates = [(score(k),k) for k in candidates ]
        candidates.sort(key=lambda s: s[0])
        return [self.extract(k) for _,k in candidates[:K] ]


    def rewriteWithInvention(self, invention, heads):
        originalHeads = heads
        inventionK = self.incorporate(invention)
        cost = self.betaLongCost([inventionK])
        heads = [ self.extractBetaLong(self.incorporate(h), cost) for h in heads ]

        v = RewriteWithInventionVisitor(invention)
        heads = [ v.execute(h) or o for h, o in zip(heads,originalHeads) ]
        return v.invention, heads

    def addInventionToGrammar(self, invention, g, frontiers, pseudoCounts=1.):
        heads = list({ e.program for f in frontiers for e in f })
        for h in heads:
            eprint(f"(old) PROGRAM {h}")
        invention, newHeads = self.rewriteWithInvention(invention, heads)
        eprint(f"INVENTION {invention}")
        for h in newHeads:
            eprint(f"(new) PROGRAM {h}")
        sourceUpdate = dict(zip(heads, newHeads))
        frontiers = [ Frontier([ FrontierEntry(program=sourceUpdate[e.program],
                                               logLikelihood=e.logLikelihood,
                                               logPrior=0.)
                                 for e in f ],
                               task=f.task)
                      for f in frontiers ]
        
        g = Grammar.uniform([invention] + g.primitives).insideOutside(frontiers, pseudoCounts)
        frontiers = [g.rescoreFrontier(f) for f in frontiers]
        return g, frontiers

        

class CloseInventionVisitor():
    """normalize free variables - e.g., if $1 & $3 occur free then rename them to $0, $1
    then wrap in enough lambdas so that there are no free variables and finally wrap in invention"""
    def __init__(self, p):
        self.p = p
        freeVariables = list(sorted(set(p.freeVariables())))
        self.mapping = {fv: j for j,fv in enumerate(freeVariables) }
    def index(self, e, d):
        if e.i - d in self.mapping:
            return Index(self.mapping[e.i - d] + d)
        return e
    def abstraction(self, e, d):
        return Abstraction(e.body.visit(self, d + 1))
    def application(self, e, d):
        return Application(e.f.visit(self, d),
                           e.x.visit(self, d))
    def primitive(self, e, d): return e
    def invented(self, e, d): return e

    def execute(self):
        normed = self.p.visit(self, 0)
        closed = normed
        for _ in range(len(self.mapping)):
            closed = Abstraction(closed)
        return Invented(closed)
        
        
class RewriteWithInventionVisitor():
    def __init__(self, p):
        v = CloseInventionVisitor(p)
        self.original = p
        self.mapping = { j: fv for fv, j in v.mapping.items() }
        self.invention = v.execute()

        self.appliedInvention = self.invention
        for j in range(len(self.mapping) - 1, -1, -1):
            self.appliedInvention = Application(self.appliedInvention, Index(self.mapping[j]))
                

    def tryRewrite(self, e):
        if e == self.original:
            application = self.invention
            for j in range(len(self.mapping) - 1, -1, -1):
                application = Application(application, Index(self.mapping[j]))
            eprint(f"REWRITE {e} > {application}")
            return application
        return None

    def index(self, e): return e
    def primitive(self, e): return e
    def invented(self, e): return e
    def abstraction(self, e):
        return self.tryRewrite(e) or Abstraction(e.body.visit(self))
    def application(self, e):
        return self.tryRewrite(e) or Application(e.f.visit(self),
                                                 e.x.visit(self))
    def execute(self, e):
        try:
            i = e.visit(self)
            l = EtaLongVisitor().execute(i)
            eprint(f"(rwi) {e} > {i} > {l}")
            return l
        except EtaExpandFailure:
            eprint(f"(rwi: failure) {e}")
            return None




            
