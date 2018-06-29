from betaExpansion import *

from program import *

def getOne(s):
    return next(iter(s))

class Union(Program):
    def __init__(self, elements):
        self.elements = frozenset(elements)
        
    @property
    def isUnion(self): return True
    def __eq__(self,o):
        return isinstance(o,Union) and self.elements == o.elements
    def __hash__(self): return hash(self.elements)
    def __str__(self):
        return "{%s}"%(", ".join(map(str,list(self.elements))))
    def __repr__(self): return str(self)
    def __iter__(self): return iter(self.elements)

class VersionTable():
    def __init__(self):
        self.expressions = []
        self.recursiveTable = []
        self.substitutionTable = {}
        self.expression2index = {}

        self.universe = self.incorporate(Primitive("U",None,None))
        self.empty = self.incorporate(Union([]))
        

    def incorporate(self,p):
        if p.isIndex or p.isPrimitive or p.isInvented:
            pass
        elif p.isAbstraction:
            p = Abstraction(self.incorporate(p.body))
        elif p.isApplication:
            p = Application(self.incorporate(p.f),
                            self.incorporate(p.x))
        elif p.isUnion:
            p = Union([self.incorporate(e) for e in p ])
        else: assert False

        return self._incorporate(p)

    def _incorporate(self,p):
        if p in self.expression2index: return self.expression2index[p]

        j = len(self.expressions)
        
        self.expressions.append(p)
        self.expression2index[p] = j
        self.recursiveTable.append(None)
        
        return j

    def extract(self,j):
        l = self.expressions[j]
        if l.isAbstraction:
            for b in self.extract(l.body):
                yield Abstraction(b)
        elif l.isApplication:
            for f in self.extract(l.f):
                for x in self.extract(l.x):
                    yield Application(f,x)
        elif l.isIndex or l.isPrimitive or l.isInvented:
            yield l
        elif l.isUnion:
            for e in l:
                yield from self.extract(e)
        else: assert False

    def reachable(self, heads):
        visited = set()
        def visit(j):
            if j in visited: return
            visited.add(j)

            l = self.expressions[j]
            if l.isUnion:
                for e in l:
                    visit(e)
            elif l.isAbstraction: visit(l.body)
            elif l.isApplication:
                visit(l.f)
                visit(l.x)

        for h in heads:
            visit(h)
        return visited

    def size(self,j):
        l = self.expressions[j]
        if l.isApplication:
            return self.size(l.f) + self.size(l.x)
        elif l.isAbstraction:
            return self.size(l.body)
        elif l.isUnion:
            return sum(self.size(e) for e in l )
        else:
            return 1
            

    def union(self,elements):
        if self.universe in elements: return self.universe
        
        _e = []
        for e in elements:
            if self.expressions[e].isUnion:
                for j in self.expressions[e]:
                    _e.append(j)
            elif e != self.empty:
                _e.append(e)

        elements = frozenset(_e)
        if len(elements) == 0: return self.empty
        if len(elements) == 1: return next(iter(elements))
        return self._incorporate(Union(elements))
    def apply(self,f,x):
        if f == self.empty: return f
        if x == self.empty: return x
        return self._incorporate(Application(f,x))
    def abstract(self,b):
        if b == self.empty: return self.empty
        return self._incorporate(Abstraction(b))
    def index(self,i):
        return self._incorporate(Index(i))

    def intersection(self,a,b):
        if a == self.empty or b == self.empty: return self.empty
        if a == self.universe: return b
        if b == self.universe: return a
        if a == b: return a

        x = self.expressions[a]
        y = self.expressions[b]

        if x.isAbstraction and y.isAbstraction:
            return self.abstract(self.intersection(x.body,y.body))
        if x.isApplication and y.isApplication:
            return self.apply(self.intersection(x.f,y.f),
                              self.intersection(x.x,y.x))
        if x.isUnion:
            if y.isUnion:
                return self.union([ self.intersection(x_,y_)
                                    for x_ in x
                                    for y_ in y ])
            return self.union([ self.intersection(x_, b)
                                for x_ in x ])
        if y.isUnion:
            return self.union([ self.intersection(a, y_)
                                for y_ in y ])
        return self.empty

    def shift(self,j,n,c=0):
        if n == 0: return j

        l = self.expressions[j]

        if l.isUnion:
            return self.union([ self.shift(e,n,c)
                                for e in l ])
        if l.isApplication:
            return self.apply(self.shift(l.f,n,c),self.shift(l.x,n,c))
        if l.isAbstraction:
            return self.abstract(self.shift(l.body,n,c+1))
        if l.isIndex:
            if l.i >= c:
                if l.i + n >= 0:
                    return self.index(l.i + n)
                else:
                    return self.empty
            return j
        assert l.isPrimitive or l.isInvented
        return j        

    def substitutions(self,j,n):
        if (j,n) in self.substitutionTable: return self.substitutionTable[(j,n)]
        
        s = self.shift(j,-n)
        if s == self.empty: m = {}
        else: m = {s: self.index(n)}

        l = self.expressions[j]
        if l.isPrimitive or l.isInvented:
            m[self.universe] = j
        elif l.isIndex:
            m[self.universe] = j if l.i < n else self.index(l.i + 1)
        elif l.isAbstraction:
            for v,b in self.substitutions(l.body, n + 1).items():
                m[v] = self.abstract(b)
        elif l.isApplication:
            newMapping = {}
            fm = self.substitutions(l.f,n)
            xm = self.substitutions(l.x,n)
            for v1,f in fm.items():
                for v2,x in xm.items():
                    v = self.intersection(v1,v2)
                    if v == self.empty: continue
                    a = self.apply(f,x)
                    if v in newMapping:
                        newMapping[v].append(a)
                    else:
                        newMapping[v] = [a]
            for v in newMapping:
                newMapping[v] = self.union(newMapping[v])
            newMapping.update(m)
            m = newMapping
        elif l.isUnion:
            newMapping = {}
            for e in l:
                for v,b in self.substitutions(e,n).items():
                    if v in newMapping:
                        newMapping[v].append(b)
                    else:
                        newMapping[v] = [b]
            for v in newMapping:
                newMapping[v] = self.union(newMapping[v])
            newMapping.update(m)
            m = newMapping
        else: assert False

        self.substitutionTable[(j,n)] = m

        return m

    def inversion(self,j):
        return self.union([self.apply(self.abstract(b),v)
                           for v,b in self.substitutions(j,0).items()
                           if v != self.universe])

    def recursiveInversion(self,j):
        if self.recursiveTable[j] is not None: return self.recursiveTable[j]
        
        l = self.expressions[j]
        if l.isUnion:
            return self.union([self.recursiveInversion(e) for e in l ])
        
        t = [self.apply(self.abstract(b),v)
             for v,b in self.substitutions(j,0).items()
             if v != self.universe]

        if l.isApplication:
            t.append(self.apply(self.recursiveInversion(l.f),l.x))
            t.append(self.apply(l.f,self.recursiveInversion(l.x)))
        elif l.isAbstraction:
            t.append(self.abstract(self.recursiveInversion(l.body)))

        ru = self.union(t)
        self.recursiveTable[j] = ru
        return ru

    def repeatedExpansion(self,j,n):
        spaces = [j]
        for _ in range(n):
            spaces.append(self.recursiveInversion(spaces[-1]))
        return spaces #self.union(spaces)
            
    def rewriteReachable(self,heads,n):
        vertices = self.reachable(heads)
        spaces = {v: self.repeatedExpansion(v,n)
                  for v in vertices }
        return spaces
            
    def loadEquivalences(self, g, spaces):
        versionClasses = [None]*len(self.expressions)
        def extract(j):
            if versionClasses[j] is not None:
                return versionClasses[j]
            
            l = self.expressions[j]
            if l.isAbstraction:
                ks = g.setOfClasses(g.abstractClass(b)
                                    for b in extract(l.body))
            elif l.isApplication:
                fs = extract(l.f)
                xs = extract(l.x)
                ks = g.setOfClasses(g.applyClass(f,x)
                                    for x in xs for f in fs )
            elif l.isUnion:
                ks = g.setOfClasses(e for u in l for e in extract(u))
            else:
                ks = g.setOfClasses({g.incorporate(l)})
            versionClasses[j] = ks
            return ks
            

        N = len(next(iter(spaces.values())))
        vertices = list(sorted(spaces.keys(), key=lambda v: self.size(v)))

        for n in range(N):
            print(f"Processing rewrites {n} steps away from original expressions...")
            for v in vertices:
                expressions = list(self.extract(v))
                assert len(expressions) == 1
                expression = expressions[0]
                k = g.incorporate(expression)
                for e in list(extract(spaces[v][n])):
                    g.makeEquivalent(k,e)

    def makeEquivalenceGraph(self,heads,n):
        #from compressionGraph import ExpressionGraph
        g = EquivalenceGraph()
        with timing("calculated version spaces"):
            spaces = self.rewriteReachable(heads,n)
        with timing("loaded equivalences"):
            self.loadEquivalences(g,spaces)
        print(f"{len(g.incident)} E nodes, {len(g.classes)} L nodes in equivalence graph.")
        print(f"{len(self.expressions)} distinct version spaces enumerated.")
        # g.visualize(simplify=False)
            
            
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
        

if __name__ == "__main__":
    from arithmeticPrimitives import *
    from listPrimitives import *
    from grammar import *
    bootstrapTarget_extra()
    p1 = Program.parse("(lambda (fold empty $0 (lambda (lambda (cons (- $0 5) $1)))))")
    #p1 = Program.parse("(lambda (fold $0  fold))")
    p2 = Program.parse("(lambda (fold empty $0 (lambda (lambda (cons (+ $0 $0) $1)))))")

    N=4
    

    v = VersionTable()
    v.makeEquivalenceGraph({v.incorporate(p1)},N)
    # with timing("calculated table space"):
    #     j = v.rewriteReachable({v.incorporate(p1)},N)
    # with timing("denotation of table space"):
    #     t = set(v.extract(j))
    
    
    # with timing("did the brute force thing"):
    #     gt = set(recursiveBetaExpand(p1,N=N))
    # vs = t
    # print(vs - gt)
    # print(gt - vs)
    # for spurious in vs - gt:
    #     print("spurious")
    #     print(spurious)
    #     print(spurious.betaNormalForm())
        
