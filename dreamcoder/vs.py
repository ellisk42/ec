from dreamcoder.grammar import *

epsilon = 0.001


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
    def __init__(self, elements, canBeEmpty=False):
        self.elements = frozenset(elements)
        if not canBeEmpty: assert len(self.elements) > 1
        
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
    def __init__(self, typed=True, identity=True, factored=False):
        self.factored = factored
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
        # Table containing (minimum cost, set of minimum cost programs)
        self.inhabitantTable = []
        # Table containing (minimum cost, set of minimum cost programs NOT starting w/ abstraction)
        self.functionInhabitantTable = []
        self.superCache = {}

        self.overlapTable = {}
        
        self.universe = self.incorporate(Primitive("U",t0,None))
        self.empty = self.incorporate(Union([], canBeEmpty=True))

    def __len__(self): return len(self.expressions)

    def clearOverlapTable(self):
        self.overlapTable = {}

    def visualize(self, j):
        from graphviz import Digraph
        g = Digraph()

        visited = set()
        def walk(i):
            if i in visited: return

            if i == self.universe:
                g.node(str(i), 'universe')
            elif i == self.empty:
                g.node(str(i), 'nil')
            else:
                l = self.expressions[i]
                if l.isIndex or l.isPrimitive or l.isInvented:
                    g.node(str(i), str(l))
                elif l.isAbstraction:
                    g.node(str(i), "lambda")
                    walk(l.body)
                    g.edge(str(i), str(l.body))
                elif l.isApplication:
                    g.node(str(i), "@")
                    walk(l.f)
                    walk(l.x)
                    g.edge(str(i), str(l.f), label='f')
                    g.edge(str(i), str(l.x), label='x')
                elif l.isUnion:
                    g.node(str(i), "U")
                    for c in l:
                        walk(c)
                        g.edge(str(i), str(c))
                else:
                    assert False
            visited.add(i)
        walk(j)
        g.render(view=True)

    def branchingFactor(self,j):
        l = self.expressions[j]
        if l.isApplication: return max(self.branchingFactor(l.f),
                                       self.branchingFactor(l.x))
        if l.isUnion: return max([len(l.elements)] + [self.branchingFactor(e) for e in l ])
        if l.isAbstraction: return self.branchingFactor(l.body)
        return 0
            
        
    def intention(self,j, isFunction=False):
        l = self.expressions[j]
        if l.isIndex or l.isPrimitive or l.isInvented: return l
        if l.isAbstraction: return Abstraction(self.intention(l.body))
        if l.isApplication: return Application(self.intention(l.f),
                                               self.intention(l.x))
        if l.isUnion: return Union(self.intention(e)
                                   for e in l )
        assert False

    def walk(self,j):
        """yields every subversion space of j"""
        visited = set()
        def r(n):
            if n in visited: return
            visited.add(n)
            l = self.expressions[n]
            yield l
            if l.isApplication:
                yield from r(l.f)
                yield from r(l.x)
            if l.isAbstraction:
                yield from r(l.body)
            if l.isUnion:
                for e in l:
                    yield from r(e)
        yield from r(j)

                
    def incorporate(self,p):
        #assert isinstance(p,Union)# or p.wellTyped()
        if p.isIndex or p.isPrimitive or p.isInvented:
            pass
        elif p.isAbstraction:
            p = Abstraction(self.incorporate(p.body))
        elif p.isApplication:
            p = Application(self.incorporate(p.f),
                            self.incorporate(p.x))
        elif p.isUnion:
            if len(p.elements) > 0:
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
        self.inhabitantTable.append(None)
        self.functionInhabitantTable.append(None)
        
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

    def haveOverlap(self,a,b):
        if a == self.empty or b == self.empty: return False
        if a == self.universe: return True
        if b == self.universe: return True
        if a == b: return True

        if a in self.overlapTable:
            if b in self.overlapTable[a]:
                return self.overlapTable[a][b]
        else: self.overlapTable[a] = {}

        x = self.expressions[a]
        y = self.expressions[b]

        if x.isAbstraction and y.isAbstraction:
            overlap = self.haveOverlap(x.body,y.body)
        elif x.isApplication and y.isApplication:
            overlap = self.haveOverlap(x.f,y.f) and \
                self.haveOverlap(x.x,y.x)
        elif x.isUnion:
            if y.isUnion:
                overlap = any( self.haveOverlap(x_,y_)
                            for x_ in x
                            for y_ in y )
            overlap = any( self.haveOverlap(x_, b)
                        for x_ in x )
        elif y.isUnion:
            overlap = any( self.haveOverlap(a, y_)
                        for y_ in y )
        else:
            overlap = False
        self.overlapTable[a][b] = overlap
        return overlap

    def minimalInhabitants(self,j):
        """Returns (minimal size, set of singleton version spaces)"""
        assert isinstance(j,int)
        if self.inhabitantTable[j] is not None: return self.inhabitantTable[j]
        e = self.expressions[j]
        if e.isAbstraction:
            cost, members = self.minimalInhabitants(e.body)
            cost = cost + epsilon
            members = {self.abstract(m) for m in members}
        elif e.isApplication:
            fc, fm = self.minimalFunctionInhabitants(e.f)
            xc, xm = self.minimalInhabitants(e.x)
            cost = fc + xc + epsilon
            members = {self.apply(f_,x_)
                       for f_ in fm for x_ in xm }
        elif e.isUnion:
            children = [self.minimalInhabitants(z)
                        for z in e ]
            cost = min(c for c,_ in children)
            members = {zp
                       for c,z in children
                       if c == cost
                       for zp in z }
        else:
            assert e.isIndex or e.isInvented or e.isPrimitive
            cost = 1
            members = {j}


        # if len(members) > 1:
        #     for m in members: break
        #     members = {m}
        self.inhabitantTable[j] = (cost, members)
        
        return cost, members

    def minimalFunctionInhabitants(self,j):
        """Returns (minimal size, set of singleton version spaces)"""
        assert isinstance(j,int)
        if self.functionInhabitantTable[j] is not None: return self.functionInhabitantTable[j]
        e = self.expressions[j]
        if e.isAbstraction:
            cost = POSITIVEINFINITY
            members = set()
        elif e.isApplication:
            fc, fm = self.minimalFunctionInhabitants(e.f)
            xc, xm = self.minimalInhabitants(e.x)
            cost = fc + xc + epsilon
            members = {self.apply(f_,x_)
                       for f_ in fm for x_ in xm }
        elif e.isUnion:
            children = [self.minimalFunctionInhabitants(z)
                        for z in e ]
            cost = min(c for c,_ in children)
            members = {zp
                       for c,z in children
                       if c == cost
                       for zp in z }
        else:
            assert e.isIndex or e.isInvented or e.isPrimitive
            cost = 1
            members = {j}

        # if len(members) > 1:
        #     for m in members: break
        #     members = {m}
            
        self.functionInhabitantTable[j] = (cost, members)
        return cost, members

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
                                                   "shiftFree_%d: %s"%(n,set(self.extract(s)))
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
        elif l.isApplication and not self.factored:
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
            # print(f"substitutions: |{len(fm)}|x|{len(xm)}| = {len(m)}\t{len(m) <= len(fm)+len(xm)}")
        elif l.isApplication and self.factored:
            newMapping = {}
            fm = self._substitutions(l.f,n)
            xm = self._substitutions(l.x,n)
            for v1,f in fm.items():
                if self.typed: v1,nType1 = v1
                for v2,x in xm.items():
                    if self.typed: v2,nType2 = v2
                    v = self.intersection(v1,v2)
                    if v == self.empty: continue
                    if v in newMapping:
                        newMapping[v] = ({f} | newMapping[v][0],
                                         {x} | newMapping[v][1])
                    else:
                        newMapping[v] = ({f},{x})
            for v,(fs,xs) in newMapping.items():
                fs = self.union(list(fs))
                xs = self.union(list(xs))
                m[v] = self.apply(fs,xs)
            # print(f"substitutions: |{len(fm)}|x|{len(xm)}| = {len(m)}\t{len(m) <= len(fm)+len(xm)}")
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
                    # print(f"\t{e.betaNormalForm()} : {e.betaNormalForm().infer()}")
                    assert n is None or e.betaNormalForm() == n
                    n = e.betaNormalForm()
                    print("the rewritten extension is")
                for e in self.extract(i):
                    print(e, e.infer())
                    # print(f"\t{e.betaNormalForm()} : {e.betaNormalForm().infer()}")
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
                    # print(f"\t{e.betaNormalForm()} : {e.betaNormalForm().infer()}")
                    assert n is None or e.betaNormalForm() == n
                    n = e.betaNormalForm()
                print("the rewritten extension is")
                for e in self.extract(ru):
                    print(e, e.infer())
                    # print(f"\t{e.betaNormalForm()} : {e.betaNormalForm().infer()}")
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

    def properVersionSpace(self, j, n):
        return self.union(self.repeatedExpansion(j, n))

    def superVersionSpace(self, j, n):
        """Construct decorated tree and then merge version spaces with subtrees via union operator"""
        if j in self.superCache: return self.superCache[j]
        spaces = self.rewriteReachable({j}, n)
        def superSpace(i):
            assert i in spaces
            e = self.expressions[i]
            components = [i] + spaces[i]
            if e.isIndex or e.isPrimitive or e.isInvented:
                pass
            elif e.isAbstraction:
                components.append(self.abstract(superSpace(e.body)))
            elif e.isApplication:
                components.append(self.apply(superSpace(e.f), superSpace(e.x)))
            elif e.isUnion: assert False
            else: assert False
            
            return self.union(components)
        self.superCache[j] = superSpace(j)
        return self.superCache[j]
            
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
            # print(f"Processing rewrites {n} steps away from original expressions...")
            for v in vertices:
                expressions = list(self.extract(v))
                assert len(expressions) == 1
                expression = expressions[0]
                k = g.incorporate(expression)
                if k is None: continue
                t0 = g.typeOfClass[k]
                if t0 not in typedClassesOfVertex[v]:
                    typedClassesOfVertex[v][t0] = k
                extracted = list(extract(spaces[v][n]))
                for e in extracted:
                    t = g.typeOfClass[e]
                    if t in typedClassesOfVertex[v]:
                        g.makeEquivalent(typedClassesOfVertex[v][t],e)
                    else:
                        typedClassesOfVertex[v][e] = e

    def bestInventions(self, versions, bs=25):
        """versions: [[version index]]"""
        """bs: beam size"""
        """returns: list of (indices to) candidates"""
        import gc
        
        def nontrivial(proposal):
            primitives = 0
            collisions = 0
            indices = set()
            for d, tree in proposal.walk():
                if tree.isPrimitive or tree.isInvented: primitives += 1
                elif tree.isIndex:
                    i = tree.i - d
                    if i in indices: collisions += 1
                    indices.add(i)
            return primitives > 1 or (primitives == 1 and collisions > 0)

        with timing("calculated candidates from version space"):
            candidates = [{j
                           for k in self.reachable(hs)
                           for _,js in [self.minimalInhabitants(k), self.minimalFunctionInhabitants(k)]
                           for j in js }
                          for hs in versions]
            from collections import Counter
            candidates = Counter(k for ks in candidates for k in ks)
            candidates = {k for k,f in candidates.items() if f >= 2 and nontrivial(next(self.extract(k))) }
            # candidates = [k for k in candidates if next(self.extract(k)).isBetaLong()]
            eprint(len(candidates),"candidates from version space")

            # Calculate the number of free variables for each candidate invention
            # This is important because, if a candidate has free variables,
            # then whenever we use it we will have to apply it to those free variables;
            # thus using a candidate with free variables is more expensive
            candidateCost = {k: len(set(next(self.extract(k)).freeVariables())) + 1
                             for k in candidates }

        inhabitTable = self.inhabitantTable
        functionTable = self.functionInhabitantTable

        class B():
            def __init__(self, j):
                cost, inhabitants = inhabitTable[j]
                functionCost, functionInhabitants = functionTable[j]
                self.relativeCost = {inhabitant: candidateCost[inhabitant]
                                     for inhabitant in inhabitants
                                     if inhabitant in candidates}
                self.relativeFunctionCost = {inhabitant: candidateCost[inhabitant]
                                             # INTENTIONALLY, do not use function inhabitants
                                             for inhabitant in inhabitants
                                             if inhabitant in candidates}
                self.defaultCost = cost
                self.defaultFunctionCost = functionCost

            @property
            def domain(self):
                return set(self.relativeCost.keys())
            @property
            def functionDomain(self):
                return set(self.relativeFunctionCost.keys())
            def restrict(self):
                if len(self.relativeCost) > bs:
                    self.relativeCost = dict(sorted(self.relativeCost.items(),
                                                    key=lambda rk: rk[1])[:bs])
                if len(self.relativeFunctionCost) > bs:
                    self.relativeFunctionCost = dict(sorted(self.relativeFunctionCost.items(),
                                                            key=lambda rk: rk[1])[:bs])
            def getCost(self, given):
                return self.relativeCost.get(given, self.defaultCost)
            def getFunctionCost(self, given):
                return self.relativeFunctionCost.get(given, self.defaultFunctionCost)
            def relax(self, given, cost):
                self.relativeCost[given] = min(cost,
                                               self.getCost(given))
            def relaxFunction(self, given, cost):
                self.relativeFunctionCost[given] = min(cost,
                                                       self.getFunctionCost(given))

            def unobject(self):
                return {'relativeCost': self.relativeCost, 'defaultCost': self.defaultCost,
                        'relativeFunctionCost': self.relativeFunctionCost, 'defaultFunctionCost': self.defaultFunctionCost}

        beamTable = [None]*len(self.expressions)

        def costs(j):
            if beamTable[j] is not None:
                return beamTable[j]

            beamTable[j] = B(j)
            
            e = self.expressions[j]
            if e.isIndex or e.isPrimitive or e.isInvented:
                pass
            elif e.isAbstraction:
                b = costs(e.body)
                for i,c in b.relativeCost.items():
                    beamTable[j].relax(i, c + epsilon)
            elif e.isApplication:
                f = costs(e.f)
                x = costs(e.x)
                for i in f.functionDomain | x.domain:
                    beamTable[j].relax(i, f.getFunctionCost(i) + x.getCost(i) + epsilon)
                    beamTable[j].relaxFunction(i, f.getFunctionCost(i) + x.getCost(i) + epsilon)
            elif e.isUnion:
                for z in e:
                    cz = costs(z)
                    for i,c in cz.relativeCost.items(): beamTable[j].relax(i, c)
                    for i,c in cz.relativeFunctionCost.items(): beamTable[j].relaxFunction(i, c)
            else: assert False

            beamTable[j].restrict()
            return beamTable[j]

        with timing("beamed version spaces"):
            beams = parallelMap(numberOfCPUs(),
                                lambda hs: [ costs(h).unobject() for h in hs ],
                                versions,
                                memorySensitive=True,
                                chunksize=1,
                                maxtasksperchild=1)

        # This can get pretty memory intensive - clean up the garbage
        beamTable = None
        gc.collect()
        
        candidates = {d
                      for _bs in beams
                      for b in _bs
                      for d in b['relativeCost'].keys() }
        def score(candidate):
            return sum(min(min(b['relativeCost'].get(candidate, b['defaultCost']),
                               b['relativeFunctionCost'].get(candidate, b['defaultFunctionCost']))
                           for b in _bs )
                       for _bs in beams )
        candidates = sorted(candidates, key=score)
        return candidates

    def rewriteWithInvention(self, i, js):
        """Rewrites list of indices in beta long form using invention"""
        self.clearOverlapTable()
        class RW():
            """rewritten cost/expression either as a function or argument"""            
            def __init__(self, f,fc,a,ac):
                assert not (fc < ac)
                self.f, self.fc, self.a, self.ac = f,fc,a,ac
        
        _i = list(self.extract(i))
        assert len(_i) == 1
        _i = _i[0]
        
        table = {}
        def rewrite(j):
            if j in table: return table[j]
            e = self.expressions[j]
            if self.haveOverlap(i, j): r = RW(fc=1,ac=1,
                                              f=_i,a=_i)
            elif e.isPrimitive or e.isInvented or e.isIndex:
                r = RW(fc=1,ac=1,
                       f=e,a=e)
            elif e.isApplication:
                f = rewrite(e.f)
                x = rewrite(e.x)
                cost = f.fc + x.ac + epsilon
                ep = Application(f.f, x.a) if cost < POSITIVEINFINITY else None
                r = RW(fc=cost, ac=cost,
                       f=ep, a=ep)
            elif e.isAbstraction:
                b = rewrite(e.body)
                cost = b.ac + epsilon
                ep = Abstraction(b.a) if cost < POSITIVEINFINITY else None
                r = RW(f=None, fc=POSITIVEINFINITY,
                       a=ep, ac=cost)
            elif e.isUnion:
                children = [rewrite(z) for z in e ]
                f,fc = min(( (child.f, child.fc) for child in children ),
                           key=cindex(1))
                a,ac = min(( (child.a, child.ac) for child in children ),
                           key=cindex(1))
                r = RW(f=f,fc=fc,
                       a=a,ac=ac)
            else: assert False
            table[j] = r
            return r
        js = [ rewrite(j).a for j in js ]
        self.clearOverlapTable()
        return js
        
    def addInventionToGrammar(self, candidate, g0, frontiers,
                              pseudoCounts=1.):
        candidateSource = next(self.extract(candidate))
        v = RewriteWithInventionVisitor(candidateSource)
        invention = v.invention

        rewriteMapping = list({e.program
                               for f in frontiers
                               for e in f })
        spaces = [self.superCache[self.incorporate(program)]
                  for program in rewriteMapping ]
        rewriteMapping = dict(zip(rewriteMapping,
                                  self.rewriteWithInvention(candidate, spaces)))

        def tryRewrite(program, request=None):
            rw = v.execute(rewriteMapping[program], request=request)
            # print(f"Rewriting {program} ({rewriteMapping[program]}) : rw={rw}")
            # print("slow-motion:")
            # try:
            #     i = rewriteMapping[program].visit(v)
            #     print(f"\ti={i}")
            #     l = EtaLongVisitor().execute(i)
            #     print(f"\tl={l}")
            # except Exception as e: print(e)
            return rw or program

        frontiers = [Frontier([FrontierEntry(program=tryRewrite(e.program, request=f.task.request),
                                             logLikelihood=e.logLikelihood,
                                             logPrior=0.)
                                       for e in f ],
                              f.task)
                     for f in frontiers ]
        # print(invention)
        # for f in frontiers: print(f.entries[0].program)
        # print()
        # print()
        g = Grammar.uniform([invention] + g0.primitives, continuationType=g0.continuationType).\
            insideOutside(frontiers,
                          pseudoCounts=pseudoCounts)
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
            return self.appliedInvention
        return None

    def index(self, e): return e
    def primitive(self, e): return e
    def invented(self, e): return e
    def abstraction(self, e):
        return self.tryRewrite(e) or Abstraction(e.body.visit(self))
    def application(self, e):
        return self.tryRewrite(e) or Application(e.f.visit(self),
                                                 e.x.visit(self))
    def execute(self, e, request=None):
        try:
            i = e.visit(self)
            l = EtaLongVisitor(request=request).execute(i)
            return l
        except (UnificationFailure, EtaExpandFailure):
            return None    
        



def induceGrammar_Beta(g0, frontiers, _=None,
                       pseudoCounts=1.,
                       a=3,
                       aic=1.,
                       topK=2,
                       topI=50,
                       structurePenalty=1.,
                       CPUs=1):
    """grammar induction using only version spaces"""
    from dreamcoder.fragmentUtilities import primitiveSize
    import gc
    
    originalFrontiers = frontiers
    frontiers = [frontier for frontier in frontiers if not frontier.empty]
    eprint("Inducing a grammar from", len(frontiers), "frontiers")

    arity = a

    def restrictFrontiers():
        return parallelMap(1,#CPUs,
                           lambda f: g0.rescoreFrontier(f).topK(topK),
                           frontiers,
                           memorySensitive=True,
                           chunksize=1,
                           maxtasksperchild=1)
    restrictedFrontiers = restrictFrontiers()
    
    def objective(g, fs):
        ll = sum(g.frontierMDL(f) for f in fs )
        sp = structurePenalty * sum(primitiveSize(p) for p in g.primitives)
        return ll - sp - aic*len(g.productions)
            
    v = None
    def scoreCandidate(candidate, currentFrontiers, currentGrammar):
        try:
            newGrammar, newFrontiers = v.addInventionToGrammar(candidate, currentGrammar, currentFrontiers,
                                                               pseudoCounts=pseudoCounts)
        except InferenceFailure:
            # And this can occur if the candidate is not well typed:
            # it is expected that this can occur;
            # in practice, it is more efficient to filter out the ill typed terms,
            # then it is to construct the version spaces so that they only contain well typed terms.
            return NEGATIVEINFINITY
            
        o = objective(newGrammar, newFrontiers)

        #eprint("+", end='')
        eprint(o,'\t',newGrammar.primitives[0],':',newGrammar.primitives[0].tp)

        # eprint(next(v.extract(candidate)))
        # for f in newFrontiers:
        #     for e in f:
        #         eprint(e.program)
        
        return o
        
    with timing("Estimated initial grammar production probabilities"):
        g0 = g0.insideOutside(restrictedFrontiers, pseudoCounts)
    oldScore = objective(g0, restrictedFrontiers)
    eprint("Starting grammar induction score",oldScore)
    
    while True:
        v = VersionTable(typed=False, identity=False)
        with timing("constructed %d-step version spaces"%arity):
            versions = [[v.superVersionSpace(v.incorporate(e.program), arity) for e in f]
                        for f in restrictedFrontiers ]
            eprint("Enumerated %d distinct version spaces"%len(v.expressions))
        
        # Bigger beam because I feel like it
        candidates = v.bestInventions(versions, bs=3*topI)[:topI]
        eprint("Only considering the top %d candidates"%len(candidates))

        # Clean caches that are no longer needed
        v.recursiveTable = [None]*len(v)
        v.inhabitantTable = [None]*len(v)
        v.functionInhabitantTable = [None]*len(v)
        v.substitutionTable = {}
        gc.collect()
        
        with timing("scored the candidate inventions"):
            scoredCandidates = parallelMap(CPUs,
                                           lambda candidate: \
                                           (candidate, scoreCandidate(candidate, restrictedFrontiers, g0)),
                                            candidates,
                                           memorySensitive=True,
                                           chunksize=1,
                                           maxtasksperchild=1)
        if len(scoredCandidates) > 0:
            bestNew, bestScore = max(scoredCandidates, key=lambda sc: sc[1])
        if len(scoredCandidates) == 0 or bestScore < oldScore:
            eprint("No improvement possible.")
            # eprint("Runner-up:")
            # eprint(next(v.extract(bestNew)))
            # Return all of the frontiers, which have now been rewritten to use the
            # new fragments
            frontiers = {f.task: f for f in frontiers}
            frontiers = [frontiers.get(f.task, f)
                         for f in originalFrontiers]
            return g0, frontiers
        
        # This is subtle: at this point we have not calculated
        # versions bases for programs outside the restricted
        # frontiers; but here we are rewriting the entire frontier in
        # terms of the new primitive. So we have to recalculate
        # version spaces for everything.
        with timing("constructed versions bases for entire frontiers"):
            for f in frontiers:
                for e in f:
                    v.superVersionSpace(v.incorporate(e.program), arity)
        newGrammar, newFrontiers = v.addInventionToGrammar(bestNew, g0, frontiers,
                                                           pseudoCounts=pseudoCounts)
        eprint("Improved score to", bestScore, "(dS =", bestScore-oldScore, ") w/ invention",newGrammar.primitives[0],":",newGrammar.primitives[0].infer())
        oldScore = bestScore

        for f in newFrontiers:
            eprint(f.summarizeFull())

        g0, frontiers = newGrammar, newFrontiers
        restrictedFrontiers = restrictFrontiers()


        
        
        
        
            
            
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
    
def testSharing(projection=2):
    
    source = "(+ 1 1)"
    N = 4 # maximum number of refactorings
    L = 6 # maximum size of expression

    # def literalSize(v,j):
    #     hs = []
    #     vp = VersionTable(typed=False)
    #     for i in v.extract(j):
    #         hs.append(vp.incorporate(i))
    #     return len(set(vp.reachable(hs)))
    
    # smart = {}
    # dumb = {}
    # for l in range(L):
    #     for n in range(N):
    #         v = VersionTable(typed=False)
            
    #         j = v.properVersionSpace(v.incorporate(Program.parse(source)),n)
    #         smart[(l,n)] = len(v.reachable({j}))
    #         dumb[(l,n)] = literalSize(v,j)
    #         print(f"vs l={l}\tn={n} sz={smart[(l,n)]}")
    #         print(f"db l={l}\tn={n} sz={dumb[(l,n)]}")
    #     # increase the size of the expression
    #     source = "(+ 1 %s)"%source
    #     print("Increased size to",l + 1)

    import numpy as np
    distinct_programs = np.zeros((L,N))
    version_size = np.zeros((L,N))
    program_memory = np.zeros((L,N))

    version_size[0,1] = 24
    distinct_programs[0,1] = 8
    program_memory[0,1] = 28
    version_size[0,2] = 155
    distinct_programs[0,2] = 63
    program_memory[0,2] = 201
    version_size[0,3] = 1126
    distinct_programs[0,3] = 534
    program_memory[0,3] = 1593
    version_size[1,1] = 48
    distinct_programs[1,1] = 24
    program_memory[1,1] = 78
    version_size[1,2] = 526
    distinct_programs[1,2] = 457
    program_memory[1,2] = 1467
    version_size[1,3] = 6639
    distinct_programs[1,3] = 8146
    program_memory[1,3] = 26458
    version_size[2,1] = 74
    distinct_programs[2,1] = 57
    program_memory[2,1] = 193
    version_size[2,2] = 1095
    distinct_programs[2,2] = 2234
    program_memory[2,2] = 7616
    version_size[2,3] = 19633
    distinct_programs[2,3] = 74571
    program_memory[2,3] = 260865
    version_size[3,1] = 101
    distinct_programs[3,1] = 123
    program_memory[3,1] = 438
    version_size[3,2] = 1751
    distinct_programs[3,2] = 9209
    program_memory[3,2] = 32931
    version_size[3,3] = 38781
    distinct_programs[3,3] = 540315
    program_memory[3,3] = 1984171
    version_size[4,1] = 129
    distinct_programs[4,1] = 254
    program_memory[4,1] = 942
    version_size[4,2] = 2488
    distinct_programs[4,2] = 35011
    program_memory[4,2] = 129513
    version_size[4,3] = 63271
    distinct_programs[4,3] = 3477046
    program_memory[4,3] = 13179440
    version_size[5,1] = 158
    distinct_programs[5,1] = 514
    program_memory[5,1] = 1962
    version_size[5,2] = 3308
    distinct_programs[5,2] = 128319
    program_memory[5,2] = 485862
    version_size[5,3] = 93400
    distinct_programs[5,3] = 21042591
    program_memory[5,3] = 81433633


    
    import matplotlib.pyplot as plot
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})

    if projection == 3:
        f = plot.figure()
        a = f.add_subplot(111, projection='3d')
        X = np.arange(0,N)
        Y = np.arange(0,L)
        X,Y = np.meshgrid(X,Y)
        Z = np.zeros((L,N))
        for l in range(L):
            for n in range(N):
                Z[l,n] = smart[(l,n)]

        a.plot_surface(X,
                       Y,
                       np.log10(Z),
                       color='blue',
                       alpha=0.3)
        for l in range(L):
            for n in range(N):
                Z[l,n] = dumb[(l,n)]


        a.plot_surface(X,
                       Y,
                       np.log10(Z),
                       color='red',
                       alpha=0.3)


    else:
        plot.figure(figsize=(3.5,3))
        plot.tight_layout()
        logarithmic = False
        if logarithmic: P = plot.semilogy
        else: P = plot.plot
        for n in range(1, 2):
            xs = np.array(range(L))*2 + 3
            P(xs,
              [version_size[l,n] for l in range(L) ],
              'purple',
              label=None if n > 1 else 'version space')
            P(xs,
              [program_memory[l,n] for l in range(L) ],
              'green',
              label=None if n > 1 else 'no version space')
            if n > 1: dy = 1
            if n == 1 and logarithmic: dy = 0.6
            if n == 1 and not logarithmic: dy = 1
            # plot.text(xs[-1], dy*version_size[L - 1,n], "n=%d"%n)
            # plot.text(xs[-1], dy*program_memory[L - 1,n], "n=%d"%n)
            
        plot.legend()
        plot.xlabel('Size of program being refactored')
        plot.ylabel('Size of VS (purple) or progs (green)')
        plot.xticks(list(xs) + [xs[-1] + 2],
                    [ str(x) if j == 0 or j == L - 1 else ''
                      for j,x in enumerate(list(xs) + [xs[-1] + 2])])
        # if not logarithmic:
        #     plot.ylim([0,100000])
                


    plot.savefig('/tmp/vs.eps')
    assert False

if __name__ == "__main__":
    
    from dreamcoder.domains.arithmetic.arithmeticPrimitives import *
    from dreamcoder.domains.list.listPrimitives import *
    from dreamcoder.fragmentGrammar import *
    bootstrapTarget_extra()
    McCarthyPrimitives()
    testSharing()

    # p = Program.parse("(#(lambda (lambda (lambda (fold $0 empty ($1 $2))))) cons (lambda (lambda (lambda ($2 (+ (+ 5 5) (+ $1 $1)) $0)))))")
    # print(EtaLongVisitor().execute(p))

    # BOOTSTRAP
    programs = [# "(lambda (fix1 $0 (lambda (lambda (if (empty? $0) 0 (+ ($1 (cdr $0)) 1))))))",
                # "(lambda (fix1 $0 (lambda (lambda (if (empty? $0) 0 (+ ($1 (cdr $0)) 1))))))",
                # "(lambda (+ $0 1))",
                # "(lambda (+ (car $0) 1))",
                # "(lambda (+ $0 (+ 1 1)))",
                # "(lambda (- $0 1))",
                # "(lambda (- $0 (+ 1 1)))",
                # "(lambda (- (car $0) 1))",
        ("(lambda (fix1 $0 (lambda (lambda (if (eq? 0 $0) empty (cons (- 0 $0) ($1 (+ 1 $0))))))))",None),
        # ("(lambda (fix1 $0 (lambda (lambda (if (empty? $0) empty (cons (cdr $0) ($1 (cdr $0))))))))",arrow(tlist(tint),tlist(tlist(tint)))),
        # drop the last element
        # ("(lambda (fix1 $0 (lambda (lambda (if (empty? (cdr $0)) empty (cons (car $0) ($1 (cdr $0))))))))",arrow(tlist(tint),tlist(tint))),
        # take in till 1
        # ("(lambda (fix1 $0 (lambda (lambda (if (eq? (car $0) 1) empty (cons (car $0) ($1 (cdr $0))))))))",arrow(tlist(tint),tlist(tint))),
                # "(lambda (lambda (fix2 $1 $0 (lambda (lambda (lambda (if (eq? $1 0) (car $0) ($2 (- $1 1) (cdr $0)))))))))",
                # "(lambda (lambda (fix2 $1 $0 (lambda (lambda (lambda (if (eq? $1 0) (car $0) ($2 (- $1 1) (cdr $0)))))))))",
                ("(lambda (fix1 $0 (lambda (lambda (if (empty? $0) 0 (+ (car $0) ($1 (cdr $0))))))))",None),
                ("(lambda (fix1 $0 (lambda (lambda (if (empty? $0) 1 (- (car $0) ($1 (cdr $0))))))))",None),
        ("(lambda (fix1 $0 (lambda (lambda (if (empty? $0) (cons 0 empty) (cons (car $0) ($1 (cdr $0))))))))",None),
                ("(lambda (fix1 $0 (lambda (lambda (if (empty? $0) (empty? empty) (if (car $0) ($1 (cdr $0)) (eq? 1 0)))))))",None),
                # "(lambda (lambda (fix2 $1 $0 (lambda (lambda (lambda (if (empty? $1) $0 (cons (car $1) ($2 (cdr $1) $0)))))))))",
        #         ("(lambda (fix1 $0 (lambda (lambda (if (empty? $0) empty (cons (+ (car $0) (car $0)) ($1 (cdr $0))))))))",None),
        #         ("(lambda (fix1 $0 (lambda (lambda (if (empty? $0) empty (cons (+ (car $0) 1) ($1 (cdr $0))))))))",None),
        # ("(lambda (fix1 $0 (lambda (lambda (if (empty? $0) empty (cons (- (car $0) 1) ($1 (cdr $0))))))))",None),
        # ("(lambda (fix1 $0 (lambda (lambda (if (empty? $0) empty (cons (cons (car $0) empty) ($1 (cdr $0))))))))",arrow(tlist(tint),tlist(tlist(tint)))),
        # ("(lambda (fix1 $0 (lambda (lambda (if (empty? $0) empty (cons (- 0 (car $0)) ($1 (cdr $0))))))))",None)
    ]
    programs = [(Program.parse(p),t) for p,t in programs ]
    N=3

    primitives = McCarthyPrimitives()
    # for p, _ in programs:
    #     for _, s in p.walk():
    #         if s.isPrimitive:
    #             primitives.add(s)
    g0 = Grammar.uniform(list(primitives))
    print(g0)

    # with timing("RUST test"):
    #     g = induceGrammar(g0, [Frontier.dummy(p, tp=tp) for p, tp in programs],
    #                   CPUs=1,
    #                   a=N,
    #                   backend="vs")
    #     eprint(g)

    # with open('vs.pickle','rb') as handle:
    #     a,kw = pickle.load(handle)
    #     induceGrammar_Beta(*a,**kw)

    with timing("induced DSL"):
        induceGrammar_Beta(g0, [Frontier.dummy(p, tp=tp) for p, tp in programs],
                           CPUs=1,
                           a=N,
                           structurePenalty=0.)

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description = "Version-space based compression")
#     parser.add_argument("--CPUs", type=int, default=1)
#     parser.add_argument("--arity", type=int, default=3)
#     parser.add_argument("--bs", type=int, default=25,
#                         help="beam size")
#     parser.add_argument("--topK", type=int, default=2)
#     parser.add_argument("--topI", type=int, default=None,
#                         help="defaults to beam size")
#     parser.add_argument("--pseudoCounts",
#                         type=float,
#                         default=1.)
#     parser.add_argument("--structurePenalty",
#                         type=float, default=1.)
#     arguments = parser.parse_args()
    
    
