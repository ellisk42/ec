from program import *

from utilities import *

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
    def __init__(self):
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

    def setOfClasses(self,ks):
        ks = {k.chase() for k in ks}
        for k in ks:
            if k not in self.externalUsers: self.externalUsers[k] = []
            self.externalUsers[k].append(ks)
        return ks

    def makeClass(self):
        k = Class()
        self.classMembers[k] = set()
        self.incident[k] = set()
        return k

    def addEdge(self,k,l):
        self.classMembers[k].add(l)
        self.classes[l].add(k)
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

    def deleteExpression(self,l):
        assert len(self.classes[l]) == 0
        if l.isApplication:
            self.incident[l.f].remove(l)
            self.incident[l.x].remove(l)
        elif l.isAbstraction:
            self.incident[l.body].remove(l)
        del self.classes[l]

    def rename(self, old, new):
        for refersToOld in list(self.classes[old]):
            self.deleteEdge(refersToOld, old)
            self.addEdge(refersToOld, new)
        self.deleteExpression(old)

        

    def incorporateClass(self,l):
        if l not in self.classes: self.classes[l] = set()
        if len(self.classes[l]) == 0:
            k = self.makeClass()
            self.addEdge(k,l)
            return k
        return getOne(self.classes[l])
    def incorporateExpression(self,l):
        if l in self.classes: return l
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
        self.incident[f].add(l)
        self.incident[x].add(l)
        return self.incorporateClass(Application(f,x))
    def abstractClass(self,b):
        b = b.chase()
        l = Abstraction(b)
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

    def minimumCosts(self, given):
        basicClasses = {k
                        for k,children in self.classMembers.items()
                        if k in given or any( l.isIndex or l.isPrimitive or l.isInvented
                                              for l in children )}
        table = {k: 1 if k in basicClasses else POSITIVEINFINITY for k in self.classMembers }

        def expressionCost(l):
            if l.isApplication: return table[l.f] + table[l.x]
            if l.isAbstraction: return table[l.body]
            if l.isPrimitive or l.isInvented: return 1
            if l.isIndex: return 1
            assert False
        def relax(e):
            old = table[e]
            new = old
            for l in self.classMembers[e]:
                new = min(expressionCost(l),new)
            return new, new < old


        q = {getOne(self.classes[i])
             for b in basicClasses
             for i in self.incident[b] }

        while True:
            if len(q) == 0:
                for k in table:
                    _,change = relax(k)
                    assert not changed
                return table
            
            n = getOne(q)
            q.remove(n)
            new, changed = relax(n)
            if changed:
                table[n] = new
                q.update(getOne(self.classes[i])
                         for i in self.incident[n])
        
        
    def extract(self,k):
        table = self.minimumCosts([])
        def expressionCost(l):
            if l.isApplication: return table[l.f] + table[l.x]
            if l.isAbstraction: return table[l.body]
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
            

    def bestInvention(self,heads):
        def score(k):
            t = self.minimumCosts([k])
            return sum(t[h] for h in heads )
        candidates = [self.reachable([h])
                      for h in heads ]
        from collections import Counter
        candidates = Counter(k for ks in candidates for k in ks)
        candidates = list({k for k,f in candidates.items() if f >= 2 })
        print(f"{len(candidates)} candidates")
        candidates = [(score(k),k) for k in candidates ]
        best = min(candidates,key = lambda s: s[0])[1]
        return self.extract(best)
