from betaExpansion import *

from program import *
from type import *



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
    def show(self, isFunction):
        return str(self)
    def __repr__(self): return str(self)
    def __iter__(self): return iter(self.elements)

class VersionTable():
    def __init__(self, typed=True, identity=True):
        self.identity = identity
        self.typed = typed
        self.debug = False
        if self.debug:
            print("WARNING: running version spaces in debug mode. Will be substantially slower.")
        
        self.expressions = []
        self.recursiveTable = []
        self.substitutionTable = {}
        self.expression2index = {}
        self.maximumShift = []
        
        self.universe = self.incorporate(Primitive("U",t0,None))
        self.empty = self.incorporate(Union([]))
        
    def intention(self,j, isFunction=False):
        l = self.expressions[j]
        if l.isIndex or l.isPrimitive or l.isInvented: return l
        if l.isAbstraction: return Abstraction(self.intention(l.body))
        if l.isApplication: return Application(self.intention(l.f),
                                               self.intention(l.x))
        if l.isUnion: return Union(self.intention(e)
                                   for e in l )
        assert False

                
    def incorporate(self,p):
        assert isinstance(p,Union) or p.wellTyped()
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

        j = self._incorporate(p)
        return j

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

    # def shift(self,j,n,c=0):
    #     if n == 0: return j

    #     l = self.expressions[j]

    #     if l.isUnion:
    #         return self.union([ self.shift(e,n,c)
    #                             for e in l ])
    #     if l.isApplication:
    #         return self.apply(self.shift(l.f,n,c),self.shift(l.x,n,c))
    #     if l.isAbstraction:
    #         return self.abstract(self.shift(l.body,n,c+1))
    #     if l.isIndex:
    #         if l.i >= c:
    #             if l.i + n >= 0:
    #                 return self.index(l.i + n)
    #             else:
    #                 return self.empty
    #         return j
    #     assert l.isPrimitive or l.isInvented
    #     return j

    def shiftFree(self,j,n,c=0):
        if n == 0: return j
        l = self.expressions[j]
        if l.isUnion:
            return self.union([ self.shiftFree(e,n,c)
                                for e in l ])
        if l.isApplication:
            return self.apply(self.shiftFree(l.f,n,c),
                              self.shiftFree(l.x,n,c))
        if l.isAbstraction:
            return self.abstract(self.shiftFree(l.body,n,c+1))
        if l.isIndex:
            if l.i < c: return j
            if l.i >= n + c: return self.index(l.i - n)
            return self.empty
        assert l.isPrimitive or l.isInvented
        return j

    def substitutions(self,j):
        if self.typed:
            for (v,_),b in self._substitutions(j,0).items():
                yield v,b
        else:
            yield from self._substitutions(j,0).items()

    def _substitutions(self,j,n):
        if (j,n) in self.substitutionTable: return self.substitutionTable[(j,n)]
        
        
        s = self.shiftFree(j,n)
        if self.debug:
            assert set(self.extract(s)) == set( e.shift(-n)
                                                for e in self.extract(j)
                                                if all( f >= n for f in e.freeVariables()  )),\
                                                   f"shiftFree_{n}: {set(self.extract(s))}"
        if s == self.empty: m = {}
        else:
            if self.typed:
                principalType = self.infer(s)
                if principalType == self.bottom:
                    print(self.infer(j))
                    print(list(self.extract(j)))
                    print(list(self.extract(s)))
                    assert False
                m = {(s, self.infer(s)[1].canonical()): self.index(n)}
            else:
                m = {s: self.index(n)}

        l = self.expressions[j]
        if l.isPrimitive or l.isInvented:
            m[(self.universe,t0) if self.typed else self.universe] = j
        elif l.isIndex:
            m[(self.universe,t0) if self.typed else self.universe] = \
                    j if l.i < n else self.index(l.i + 1)
        elif l.isAbstraction:
            for v,b in self._substitutions(l.body, n + 1).items():
                m[v] = self.abstract(b)
        elif l.isApplication:
            newMapping = {}
            fm = self._substitutions(l.f,n)
            xm = self._substitutions(l.x,n)
            for v1,f in fm.items():
                if self.typed: v1,nType1 = v1
                for v2,x in xm.items():
                    if self.typed: v2,nType2 = v2

                    a = self.apply(f,x)
                    # See if the types that they assigned to $n are consistent
                    if self.typed:
                        if self.infer(a) == self.bottom: continue
                        try:
                            nType = canonicalUnification(nType1, nType2,
                                                         self.infer(a)[0].get(n,t0))
                        except UnificationFailure:
                            continue
                        
                    v = self.intersection(v1,v2)
                    if v == self.empty: continue
                    if self.typed and self.infer(v) == self.bottom: continue

                    key = (v,nType) if self.typed else v                        
                        
                    if key in newMapping:
                        newMapping[key].append(a)
                    else:
                        newMapping[key] = [a]
            for v in newMapping:
                newMapping[v] = self.union(newMapping[v])
            newMapping.update(m)
            m = newMapping
        elif l.isUnion:
            newMapping = {}
            for e in l:
                for v,b in self._substitutions(e,n).items():
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
        i = self.union([self.apply(self.abstract(b),v)
                         for v,b in self.substitutions(j)
                         if v != self.universe])
        if self.debug and self.typed:
            if not (self.infer(i) == self.infer(j)):
                print("inversion produced space with a different type!")
                print("the original type was",self.infer(j))
                print("the type of the rewritten expressions is",self.infer(i))
                print("the original extension was")
                n = None
                for e in self.extract(j):
                    print(e, e.infer())
                    print(f"\t{e.betaNormalForm()} : {e.betaNormalForm().infer()}")
                    assert n is None or e.betaNormalForm() == n
                    n = e.betaNormalForm()
                    print("the rewritten extension is")
                for e in self.extract(i):
                    print(e, e.infer())
                    print(f"\t{e.betaNormalForm()} : {e.betaNormalForm().infer()}")
                    assert n is None or e.betaNormalForm() == n
                    assert self.infer(i) == self.infer(j)
                assert False
        return i


    def recursiveInversion(self,j):
        if self.recursiveTable[j] is not None: return self.recursiveTable[j]
        
        l = self.expressions[j]
        if l.isUnion:
            return self.union([self.recursiveInversion(e) for e in l ])
        
        t = [self.apply(self.abstract(b),v)
             for v,b in self.substitutions(j)
             if v != self.universe and (self.identity or b != self.index(0))]
        if self.debug and self.typed:
            ru = self.union(t)
            if not (self.infer(ru) == self.infer(j)):
                print("inversion produced space with a different type!")
                print("the original type was",self.infer(j))
                print("the type of the rewritten expressions is",self.infer(ru))
                print("the original extension was")
                n = None
                for e in self.extract(j):
                    print(e, e.infer())
                    print(f"\t{e.betaNormalForm()} : {e.betaNormalForm().infer()}")
                    assert n is None or e.betaNormalForm() == n
                    n = e.betaNormalForm()
                print("the rewritten extension is")
                for e in self.extract(ru):
                    print(e, e.infer())
                    print(f"\t{e.betaNormalForm()} : {e.betaNormalForm().infer()}")
                    assert n is None or e.betaNormalForm() == n
            assert self.infer(ru) == self.infer(j)


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
        return spaces
            
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

        # maps from a vertex to a map from types to classes
        # the idea is to only enforceable equivalence between terms of the same type
        typedClassesOfVertex = {v: {} for v in vertices }
        
        for n in range(N):
            print(f"Processing rewrites {n} steps away from original expressions...")
            for v in vertices:
                expressions = list(self.extract(v))
                assert len(expressions) == 1
                expression = expressions[0]
                k = g.incorporate(expression)
                if k is None: continue
                t0 = g.typeOfClass[k]
                if t0 not in typedClassesOfVertex[v]:
                    typedClassesOfVertex[v][t0] = k
                
                for e in list(extract(spaces[v][n])):
                    t = g.typeOfClass[e]
                    if t in typedClassesOfVertex[v]:
                        g.makeEquivalent(typedClassesOfVertex[v][t],e)
                    else:
                        typedClassesOfVertex[v][e] = e

    def makeEquivalenceGraph(self,heads,n):
        from eg import EquivalenceGraph
        g = EquivalenceGraph(typed=False)
        with timing("calculated version spaces"):
            spaces = self.rewriteReachable(heads,n)
        print(f"{len(self.expressions)} distinct version spaces enumerated.")
        with timing("loaded equivalences"):
            self.loadEquivalences(g,spaces)
        print(f"{len(g.incident)} E nodes, {len(g.classes)} L nodes in equivalence graph.")
        # g.visualize(simplify=False)
        return g
            
            

            
            
def testTyping(p):
    v = VersionTable()
    j = v.incorporate(p)
    
    wellTyped = set(v.extract(v.inversion(j)))
    print(len(wellTyped))
    v = VersionTable(typed=False)
    j = v.incorporate(p)
    arbitrary = set(v.extract(v.recursiveInversion(v.recursiveInversion(v.recursiveInversion(j)))))
    print(len(arbitrary))
    assert wellTyped <= arbitrary
    assert wellTyped == {e
                         for e in arbitrary if e.wellTyped() }
    assert all( e.wellTyped() for e in wellTyped  )

    import sys
    sys.exit()
    
def testSharing():
    from versionSpace import ExpressionTable
    source = "(+ 1 1)"
    N = 100
    for _ in range(N):
        t = ExpressionTable()
        t.invert(t.incorporate(Program.parse(source)))
        v = VersionTable(typed=False)
        v.inversion(v.incorporate(Program.parse(source)))
        print(len(v.expressions),len(t))
        source = f"(+ 1 {source})"
    assert False
        
if __name__ == "__main__":
    from arithmeticPrimitives import *
    from listPrimitives import *
    from grammar import *
    bootstrapTarget_extra()
    # testSharing()
    # v = VersionTable(typed=False)
    # j = v.incorporate(Program.parse("(+ 1 (+ 1 1))"))
    # print(v.intention(v.inversion(j)))
    # assert False
    p1 = Program.parse("(lambda (fold $0 empty (lambda (lambda (cons (- $1 5) $0)))))")
#    testTyping(Program.parse("((lambda $0) cons ((lambda $0) 9))"))
    p2 = Program.parse("(lambda (fold $0 empty (lambda (lambda (cons (+ $1 $1) $0)))))")

    # eprint(EtaLongVisitor().execute(Program.parse("+")))
    # assert False

    N=2
    

    v = VersionTable(typed=False, identity=True)
    v.incorporate(p1)
    g = v.makeEquivalenceGraph({v.incorporate(p1),
                                v.incorporate(p2)},
                               N)
    with timing("invented a new primitive"):
        i = g.bestInvention([g.incorporate(p1),
                             g.incorporate(p2)])
        print(g.rewriteWithInvention(i, [p1,p2]))
    
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
        
