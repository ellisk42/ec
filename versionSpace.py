from utilities import *
from program import *
from betaExpansion import *

def getOne(s):
    return next(iter(s))

class ExpressionTable():
    def __init__(self):
        self.expressions = []
        self.expression2index = {}
        self.freeVariables = []

        self.childrenTable = []
        self.substituteChildTable = {}
        self.substitutionTable = {}
        self.inversionTable = []
        self.recursiveInversionTable = []
        
        self.equivalenceClass = []
        self.uf = UnionFind()
        
        self.newEquivalences = []

        # Map from expression index to all of the indices with beta edge is pointing to it
        self.edgesTo = []
        # Map from expression index to all of the indices it beta reduces to in one step
        self.edgesFrom = []

        # Special primitive that means the set of all possible lambda expressions
        self.Omega = self.incorporate(Primitive("OMEGA",None,None))

    def addEdge(self, _=None, destination=None, source=None):
        self.edgesTo[destination].add(source)
        self.edgesFrom[source].add(destination)

    def __len__(self): return len(self.expressions)

    def incorporate(self,p):
        if p.isIndex or p.isPrimitive or p.isInvented:
            pass
        elif p.isAbstraction:
            p = Abstraction(self.incorporate(p.body))
        elif p.isApplication:
            p = Application(self.incorporate(p.f),
                            self.incorporate(p.x))
        else: assert False

        return self._incorporate(p)

    def _incorporate(self,p):
        if p in self.expression2index: return self.expression2index[p]

        if p.isIndex: free = [p.i]           
        elif p.isPrimitive or p.isInvented: free = []
        elif p.isAbstraction: free = [f - 1
                                      for f in self.freeVariables[p.body]
                                      if f > 0]
        elif p.isApplication: free = list(set(self.freeVariables[p.f] + self.freeVariables[p.x]))
        else: assert False


        j = len(self.expressions)
        
        self.expressions.append(p)
        self.expression2index[p] = j
        self.freeVariables.append(free)
        self.childrenTable.append(None)
        self.inversionTable.append(None)
        self.recursiveInversionTable.append(None)
        self.equivalenceClass.append(self.uf.newClass(j))
        self.edgesTo.append(set())
        self.edgesFrom.append(set())

        return j

    def extract(self,j):
        l = self.expressions[j]
        if l.isAbstraction:
            return Abstraction(self.extract(l.body))
        elif l.isApplication:
            return Application(self.extract(l.f),
                               self.extract(l.x))
        return l

    def equivalentK(self,k,j):
        if k == 0: return j
        elif k >= 2:
            for e in self.equivalentK(k - 1, j):
                yield from self.equivalentK(1, e)
        else:
            assert j == 1
            for e in self.edgesTo[j]:
                #yield from self.equivalentK(k - 1, e)
                yield e
            l = self.expressions[j]
            if l.isIndex or l.isPrimitive or l.isInvented:
                if k == 0: yield j
            elif l.isAbstraction:
                for b in self.equivalentK(k,l.body):
                    yield self._incorporate(Abstraction(b))
            elif l.isApplication:
                for i in range(k + 1):
                    for f in self.equivalentK(i,l.f):
                        for x in self.equivalentK(k - i,l.x):
                            yield self._incorporate(Application(f,x))
            else: assert False
            

    def shift(self,j, d, c=0):
        if d == 0:# or all( f < c for f in self.freeVariables[j] ): return j
            return j

        l = self.expressions[j]
        if l.isIndex:
            assert l.i >= c
            assert l.i + d >= 0
            return self._incorporate(Index(l.i + d))
        elif l.isAbstraction:
            return self._incorporate(Abstraction(self.shift(l.body, d, c + 1)))
        elif l.isApplication:
            return self._incorporate(Application(self.shift(l.f, d, c),
                                                 self.shift(l.x, d, c)))
        else:
            assert False
            

    
    def SC(self,n,j):
        if (n,j) in self.substituteChildTable: return self.substituteChildTable[(n,j)]
        
        e = self.expressions[j]
        mapping = {self.Omega: set()}

        if all( fv - n >= 0 for fv in self.freeVariables[j] ):
            v = self.shift(j,-n)
            mapping[v] = {self._incorporate(Index(n))}

        if e.isIndex:
            if e.i < n:
                mapping[self.Omega] = {j}
            else:
                mapping[self.Omega] = {self._incorporate(Index(e.i + 1))}
        elif e.isApplication:
            fm = self.SC(n,e.f)
            xm = self.SC(n,e.x)
            for v in fm:
                if v == self.Omega: continue
                if not (v in xm): continue
                if not (v in mapping): mapping[v] = set()
                mapping[v].update(self._incorporate(Application(fp,xp))
                                  for fp in fm[v]
                                  for xp in xm[v] )
            for ft in fm.get(self.Omega,[]):
                # ft: program
                for v,xt in ((xValue,xBody)
                           for xValue, xBodies in xm.items()
                           for xBody in xBodies):
                    if not (v in mapping): mapping[v] = set()
                    mapping[v].add(self._incorporate(Application(ft,xt)))
            for xt in xm.get(self.Omega,[]):
                # ft: program
                for v,ft in ((fValue,fBody)
                           for fValue, fBodies in fm.items()
                           for fBody in fBodies):
                    if not (v in mapping): mapping[v] = set()
                    mapping[v].add(self._incorporate(Application(ft,xt)))

        elif e.isAbstraction:
            bm = self.SC(n + 1, e.body)
            for v,bodies in bm.items():
                bodies = {self._incorporate(Abstraction(b)) for b in bodies }
                if v not in mapping:
                    mapping[v] = bodies
                else:
                    mapping[v].update(bodies)
        elif e.isPrimitive or e.isInvented:
            mapping[self.Omega].add(j)
        else: assert False

        # print(f"SC_{n}({self.extract(j)}) = { {self.extract(v): {self.extract(b) for b in bs } for v,bs in mapping.items() } }")
        # from frozendict import frozendict
        # mapping = frozendict({v: frozenset(bs) for v,bs in mapping.items() })

        self.substituteChildTable[(n,j)] = mapping

        return mapping

            
    def invertK(self,k,j):
        s = []
        for p in self.equivalentK(k - 1, j):
            for v,bodies in self.SC(0,p).items():
                if v is self.Omega: continue
                for b in bodies:
                    f = self._incorporate(Abstraction(b))
                    a = self._incorporate(Application(f,v))
                    s.append(a)
        return set(s)

    def betaExpandK(self, k, heads):
        print("Expanding:")
        toExpand = self.reachable({p
                                   for h in heads
                                   for p in self.equivalentK(k - 1, h)})
        for p in toExpand:
            print(self.extract(p))
            for v,bodies in self.SC(0,p).items():
                if v is self.Omega: continue
                for b in bodies:
                    f = self._incorporate(Abstraction(b))
                    a = self._incorporate(Application(f,v))
                    self.addEdge(source=a, destination=p)
                    print(f"\t <--- {self.extract(a)}")
            print()
        print()
    def repeatedBetaExpandK(self, K, heads):
        previous = set()
        for k in range(1, K+1):
            self.betaExpandK(k, heads)
        
                
    def visualize(self):
        from graphviz import Digraph

        d = Digraph(comment='expression graph')

        for j,l in enumerate(self.expressions):
            if l.isPrimitive or l.isIndex or l.isInvented: label = str(l)
            elif l.isAbstraction: label = "abs"
            elif l.isApplication: label = "@"
            d.node(str(j), label)

        for j,l in enumerate(self.expressions):
            if l.isApplication:
                d.edge(str(j), str(l.f), label="f")
                d.edge(str(j), str(l.x), label="x")
            elif l.isAbstraction:
                d.edge(str(j), str(l.body))

            for b in self.edgesFrom[j]:
                d.edge(str(j), str(b), label="B", color="cyan")

        d.render('/tmp/betaGraph.gv', view=True)
                

    def invert(self,j):
        if self.inversionTable[j] is not None: return self.inversionTable[j]

        s = []
        if False:
            for v,mapping in self.CC(j).items():
                for b in self.substitutions(0,v,mapping,j):
                    f = self._incorporate(Abstraction(b))
                    a = self._incorporate(Application(f,v))
                    s.append(a)
        else:
            gt = self.extract(j).betaNormalForm()
            for v,bodies in self.SC(0,j).items():
                if v is self.Omega: continue
                for b in bodies:
                    f = self._incorporate(Abstraction(b))
                    a = self._incorporate(Application(f,v))
                    s.append(a)
                    a = self.extract(a)
                    if a.betaNormalForm() != gt:
                        print(self.extract(j))
                        print(gt)
                        print(a)
                        assert False
                    
                
                
        self.inversionTable[j] = s
        return s

    def recursiveInvert(self,j):
        if self.recursiveInversionTable[j] is not None: return self.recursiveInversionTable[j]

        s = list(self.invert(j))

        l = self.expressions[j]
        if l.isApplication:
            for f in self.recursiveInvert(l.f):
                s.append(self._incorporate(Application(f,l.x)))
            for x in self.recursiveInvert(l.x):
                s.append(self._incorporate(Application(l.f,x)))                    
        elif l.isAbstraction:
            for b in self.recursiveInvert(l.body):
                s.append(self._incorporate(Abstraction(b)))
        
        self.recursiveInversionTable[j] = s
        self.newEquivalences.append((j,s))
        return s

    def expand(self,j,n=1):
        es = {j}
        self.newEquivalences = []
        previous = {j}
        for iteration in range(n):
            print(f"Starting iteration {iteration + 1}")
            previous = {rw
                        for p in previous
                        for rw in self.recursiveInvert(p) }
            if False:
                print("Enforcing equivalences")
                for oldIndex, newIndices in self.newEquivalences:
                    for n in newIndices: self.uf.unify(oldIndex,n)
            es.update(previous)
            self.newEquivalences = []
            print(f"Finished iteration {iteration + 1}")
        return es

    def minimumCosts(self, givens, alternatives):
        costTable = [None]*len(self)

        def cost(j):
            if costTable[j] is not None: return costTable[j]

            if j in givens: c = 1
            else:
                l = self.expressions[j]
                if l.isApplication:
                    c = cost(l.f) + cost(l.x)
                elif l.isAbstraction:
                    c = cost(l.body)
                elif l.isIndex or l.isPrimitive or l.isInvented:
                    c = 1
                else: assert False

            costTable[j] = c
            return c

        return sum(min(cost(a) for a in ass ) for ass in alternatives )

    def reachable(self,indices):
        visited = set()
        def visit(i):
            if i in visited: return
            visited.add(i)
            l = self.expressions[i]
            if l.isApplication:
                visit(l.f)
                visit(l.x)
            elif l.isAbstraction:
                visit(l.body)
        for j in indices:
            visit(j)
        return visited
    
    def bestInvention(self, alternatives):
        from collections import Counter
        
        candidates = [ self.reachable(alternative)
                       for alternative in alternatives ]
        candidates = Counter(k for ks in candidates for k in ks)
        candidates = {k for k,f in candidates.items() if f >= 2 }

        j = min(candidates, key = lambda candidate: self.minimumCosts({candidate},alternatives))
        return self.extract(j)

        
                    

def testVisual():
    bootstrapTarget_extra()
    p = Program.parse("(lambda (+ $0 5))")
    p = Program.parse("5")

    N = 3

    v = ExpressionTable()
    j = v.incorporate(p)
    v.repeatedBetaExpandK(N,{j})
    for k in range(0,N+1):
        print(f"{k} steps away:")
        for _e in {v.extract(e)
                   for e in v.equivalentK(k,j)}:
            print(_e)
        print()
    beta = {v.extract(e)
            for k in range(0,N+1)
            for e in v.equivalentK(k,j)}
    print(len(v))
    # v.visualize()
    

    v = ExpressionTable()
    j = v.incorporate(p)
    bruteForce = {v.extract(e) for e in v.expand(j,n=N)}
    print(len(v))
    # v.visualize()


    print("In brute force but not in beta")
    for e in bruteForce - beta:
        print(e)
        assert e.betaNormalForm() == p
        for _ in range(N):
            e = e.betaReduce()
            print(f"\t--> {e}")
        print()
              
    print("In beta but not in brute force")
    for e in beta - bruteForce:
        print(e)
        assert e.betaNormalForm() == p
        for _ in range(N):
            e = e.betaReduce()
            print(f"\t--> {e}")
        print()        
    
    assert False

def testSubstitute():
    v = ExpressionTable()
    p = Program.parse("(lambda ($0 (+ 5) $1))")
    v.SC(0,v.incorporate(p))
    assert False
        
if __name__ == "__main__":
    from arithmeticPrimitives import *
    from listPrimitives import *
    from grammar import *
    bootstrapTarget_extra()
    p1 = Program.parse("(lambda (fold empty $0 (lambda (lambda (cons (- $0 5) $1)))))")
    p2 = Program.parse("(lambda (fold empty $0 (lambda (lambda (cons (+ $0 $0) $1)))))")

    # testSubstitute()
    # testVisual()

    N = 4
    v = ExpressionTable()
    with timing("Computed expansions"):
        b1 = v.expand(v.incorporate(p1),n=N)
#        b2 = v.expand(v.incorporate(p2),n=N)
        print(f"expression table has size {len(v)}")
    with timing("invented a primitive"):
        print(v.bestInvention([b1,b2]))
