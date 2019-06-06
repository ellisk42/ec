import binutil

from lib.utilities import *

DEBUG = False

def getOne(s):
    return next(iter(s))

class Lifted():
    def __ne__(self,o):
        assert isinstance(o,Lifted)
        return not (o == self)
    @property
    def isLeaf(self): return False
    @property
    def isApplication(self): return False
    @property
    def isAbstraction(self): return False
    @property
    def isIndex(self): return False

    def __repr__(self): return str(self)

        

class LiftedLeaf(Lifted):
    def __init__(self, primitive):
        self.primitive = primitive
        self.freeVariables = frozenset()
    def __eq__(self,o): return isinstance(o,LiftedLeaf) and o.primitive == self.primitive
    def __hash__(self): return hash(self.primitive)

    @property
    def isLeaf(self): return True

    def __str__(self): return str(self.primitive)

    def chase(self): return self

    def shift(self,d,c=0): return self
    def closedChildren(self,n=0):
        yield Terms([self])
    def extension(self): yield self.primitive
    
    def substitutions(self,n,v): return self

    def recursiveInvert(self): return Terms([])
        

class LiftedApply(Lifted):
    def __init__(self,f,x):
        self.f = f
        self.x = x
        self.freeVariables = self.f.freeVariables | self.x.freeVariables
        
    def __eq__(self,o): return isinstance(o,LiftedApply) and self.f == o.f and self.x == o.x
    def __hash__(self): return hash((hash(self.f),hash(self.x)))

    @property
    def isApplication(self): return True

    def __str__(self): return "@(%s,%s)"%(str(self.f), str(self.x))

    def chase(self):
        f = self.f.chase()
        x = self.x.chase()
        if f != self.f or x != self.x:
            return LiftedApply(f,x)
        return self

    def shift(self,d,c=0):
        return LiftedApply(self.f.shift(d,c),
                           self.x.shift(d,c))
    def closedChildren(self,n=0):
        yield self.shift(-d)
        yield from self.f.closedChildren(n)
        yield from self.x.closedChildren(n)

    def extension(self):
        for f in self.f.extension():
            for x in self.x.extension():
                yield Application(f,x)

    def substitutions(self,n,v):
        return LiftedApply(self.f.substitutions(n,v),
                           self.x.substitutions(n,v))

    def recursiveInvert(self):
        return Terms([LiftedApply(self.f.recursiveInvert(),self.x),
                      LiftedApply(self.f,self.x.recursiveInvert())])
                                        
    

class LiftedAbstract(Lifted):
    def __init__(self,body):
        self.body = body
        self.freeVariables = frozenset(f - 1 for f in body.freeVariables if f > 0 )
        
    def __eq__(self,o): return isinstance(o,LiftedAbstract) and o.body == self.body
    def __hash__(self): return hash(hash(self.body))

    @property
    def isAbstraction(self): return True

    def __str__(self): return "abs(%s)"%str(self.body)

    def chase(self):
        b = self.body.chase()
        if b != self.body:
            return LiftedAbstract(b)
        return self

    def shift(self,d,c=0):
        return LiftedAbstract(self.body.shift(d,c+1))
    def closedChildren(self,n=0):
        yield from self.body.closedChildren(n+1)
    def extension(self):
        for b in self.body.extension():
            yield Abstraction(b)

    def substitutions(self,n,v):
        return LiftedAbstract(self.body.substitutions(n + 1, v))

    def recursiveInvert(self):
        return Terms([LiftedAbstract(self.body.recursiveInvert())])
        

class LiftedIndex(Lifted):
    def __init__(self,i):
        self.i = i
        self.freeVariables = frozenset([i])
    def __eq__(self,o): return isinstance(o,LiftedIndex) and o.i == self.i
    def __hash__(self): return hash(self.i)
    @property
    def isIndex(self): return True

    def __str__(self): return "$%d"%self.i

    def chase(self): return self

    def shift(self,d,c=0):
        if self.i >= c: return LiftedIndex(self.i + d)
        else: return self

    def closedChildren(self,n=0):
        return
        yield

    def extension(self): yield Index(self.i)

    def substitutions(self,n,v):
        if self.i < n: return self
        return LiftedIndex(self.i + 1)

    def recursiveInvert(self): return Terms([])

    def incorporate(self,g):
        return {g.indexClass(self.i)}
        


class Terms():
    def __init__(self, elements):
        self.elements = frozenset(elements)
        self.freeVariables = reduce(lambda x,y: x|y, map(lambda e: e.freeVariables, list(elements)), set())

    def __eq__(self,o):
        return isinstance(o,Terms) and self.elements == o.elements
    def __ne__(self,o): return not (self == o)
    def __hash__(self): return hash(self.elements)
    def __str__(self):
        return "{%s}"%(",".join(map(str,list(self.elements))))
        return "E_%d"%self.name
        return "E_%d: {%s}"%(self.name,
                             ",".join(str(fv) for fv in self.freeVariables))
    def __repr__(self): return str(self)
    def __iter__(self): return iter(self.elements)
    def __or__(self,o):
        assert isinstance(o,Terms)
        return Terms(self.elements|o.elements)

    def __le__(self,o):
        """Whether one versions space contains another"""
        # self <= o
        # Is everything  in self also inside o?
        return all( any( (t1.isIndex and t2.isIndex and t1 == t2) or \
                         (t1.isLeaf and t2.isLeaf and t1 == t2) or \
                         (t1.isAbstraction and t2.isAbstraction and t1.body <= t2.body) or \
                         (t1.isApplication and t2.isApplication and t1.f <= t2.f and t1.x <= t2.x)
                         for t2 in o)
                    for t1 in self )

    def closedChildren(self,n=0):
        if all( fv >= n for fv in self.freeVariables ):
            yield self.shift(-n)
        for e in self:
            yield from e.closedChildren(n)

    def shift(self,d,c=0):
        return Terms({t.shift(d,c) for t in self })

    def extension(self):
        for t in self:
            yield from t.extension()

    def substitutions(self,n,v):
        s = [ t.substitutions(n,v)
              for t in self ]
        if v.shift(n) <= self: s.append(LiftedIndex(n))
        return Terms(s)

    def invert(self):
        """beta expansion at the top level"""
        if self in Terms.TOPINVERSION: return Terms.TOPINVERSION[self]
        
        # i = Terms([ LiftedApply(Terms([LiftedAbstract(t.substitutions(0,v))]),v)
        #             for t in self
        #             for v in t.closedChildren()])
        i = Terms([ lift(ie)
                    for e in self.extension()
                    for ie in inverseBeta_(e) ])
        Terms.TOPINVERSION[self] = i
        return i

    def recursiveInvert(self):
        if self in Terms.RECURSIVEINVERSION: return Terms.RECURSIVEINVERSION[self]
        i = Terms(reduce(lambda x,y: x|y, [e.recursiveInvert() for e in self] + [self.invert()]))
        Terms.EQUIVALENCES.newClass(self)
        Terms.EQUIVALENCES.newClass(i)
        Terms.EQUIVALENCES.unify(self,i)
        Terms.RECURSIVEINVERSION[self] = i
        return i
    def R(self,n=1):
        if n == 1: return self.recursiveInvert()
        return self.recursiveInvert().R(n - 1)

    def incorporate(self,g):
        return { e
                 for l in self
                 for e in l.incorporate(g) }

Terms.RECURSIVEINVERSION = {}
Terms.TOPINVERSION = {}
Terms.EQUIVALENCES = UnionFind()
        
            
def lift(t):
    assert isinstance(t,Program)
    if t.isApplication:
        t = LiftedApply(lift(t.f),lift(t.x))
    elif t.isAbstraction:
        t = LiftedAbstract(lift(t.body))
    elif t.isIndex:
        t = LiftedIndex(t.i)
    else:
        t = LiftedLeaf(t)
    return Terms({t})

class EquivalenceClass():
    def __init__(self, name, freeVariables):
        self.name = name
        
        self.freeVariables = freeVariables
        self.leader = None

    def __eq__(self,o): return isinstance(o,EquivalenceClass) and self.name == o.name
    def __ne__(self,o): return not (self == o)
    def __hash__(self): return hash(self.name)
    def __str__(self):
        return "E_%d"%self.name
        return "E_%d: {%s}"%(self.name,
                             ",".join(str(fv) for fv in self.freeVariables))
    def __repr__(self): return str(self)

    def chase(self):
        k = self
        while k.leader is not None:
            k = k.leader
        return k


        

class ExpressionGraph():
    """Lifted expression: a program whose children are EquivalenceClass's"""
    def __init__(self):
        # Map from equivalence class to an immutable set of all the things in that class
        # self.classes.keys() should give all of the equivalence nodes in the graph
        self.e2l = {}

        # Map from an equivalence class to all of the lifted expressions that point to that class
        self.incident = {}

        # Map from a lifted expression to the set of equivalence classes it belongs to
        self.l2e = {}

        # Things that nothing in the graph should ever refer to
        self.graveyard = []

        # Each class is given a unique ID
        self.nextClass = 0

    def debug(self, verbose=True):
        if verbose:
            eprint()
            eprint("e2l (map from equivalence class to {lifted})")
            for e,ls in self.e2l.items():
                eprint("E_%d = {%s}"%(e.name,
                                     ",".join(map(str,list(ls)))))
            eprint()
            eprint("incident (maps equivalent class to everything that points to that class)")
            for e,ls in self.incident.items():
                eprint("{%s} -> E_%d"%(",".join(map(str,list(ls))),e.name))
            eprint()
            eprint("l2e (maps LE to {EK})")
            for l,es in self.l2e.items():
                eprint("%s ~ {%s}"%(l,",".join(map(str,list(es)))))
            eprint()
            eprint("Graveyard (equivalence classtes)")
            eprint(", ".join(map(str, (e for e in self.graveyard if isinstance(e,EquivalenceClass) ))))
            eprint()
            eprint("Graveyard (lifted expressions)")
            eprint(", ".join(map(str, (e for e in self.graveyard if isinstance(e,Lifted) ))))
            eprint()
            eprint()
        with timing("Debugged expression graph"):
            eprint("Checking consistency of equivalence classes")
            classes = frozenset(self.e2l.keys())
            assert frozenset(self.incident.keys()) == classes
            assert frozenset(e for es in self.l2e.values() for e in es ) <= classes

            eprint("Checking consistency of lifted expressions")
            lift = frozenset(self.l2e.keys())
            assert frozenset(l for ls in self.incident.values() for l in ls ) <= lift
            assert frozenset(l for ls in self.e2l.values() for l in ls ) == lift

            eprint("Checking graveyard consistency")
            for l in lift:
                assert l not in self.graveyard
            for e in classes:
                assert e not in self.graveyard

            # Make sure that all of the equivalence classes are disjoint
            # eprint("Checking for disjointed")
            
            # used = set()
            # for e in classes:
            #     new = self.e2l[e]
            #     assert len(used & new) == 0
            # used = used|new

            eprint("Checking consistency of incidents")
            # Make sure that everything that the equivalence classes claim to use actually use them
            for e in classes:
                for u in self.incident[e]:
                    if u.isAbstraction:
                        assert u.body == e
                    elif u.isApplication:
                        assert u.f == e or u.x == e
                    else: assert False

                incident = { l
                             for l in lift
                             if (l.isAbstraction and l.body == e) or (l.isApplication and (l.f == e or l.x == e))}
                assert len(incident^self.incident[e]) == 0

    
    def extract(self,eq):
        """Yields a stream of increasingly large (acyclic) expressions
        drawn from the equivalence class"""
        eq = eq.chase()
        q = PQ()
        
        def choice(cost, xs):
            for x in xs:
                q.push(-cost, x)


        # continuation will be called with (size, expression)
        
        def visitEquivalent(cost, visited, e, k):
            if e in visited: return
            visited = visited|{e}
            # eprint("Visiting equivalence class",e,"with cost-so-far",cost)
            choice(cost + 1, list(map(lambda element: lambda: visitLifted(cost, visited, element, k),
                                      self.children(e))))
        def visitLifted(cost, visited, l, k):
            # eprint("Visiting lifted expression",l,"with cost-so-far",cost)
            if l.isAbstraction:
                visitEquivalent(cost + 1,
                                visited,
                                l.body,
                                lambda bodySize,actualBody: k(1 + bodySize, Abstraction(actualBody)))
            elif l.isApplication:
                visitEquivalent(cost + 1,
                                visited,
                                l.f,
                                lambda fSize, f: visitEquivalent(cost + 1 + fSize,
                                                                 visited,
                                                                 l.x,
                                                                 lambda xSize, x: k(1 + fSize + xSize,
                                                                                    Application(f,x))))
            elif l.isLeaf:
                k(1,l.primitive)
            elif l.isIndex:
                k(1,Index(l.i))
            else: assert False


        repository = []
        visitEquivalent(0,set(), eq,lambda size, expression: repository.append((size, expression)))

        while len(q) > 0:
            nextContinuation = q.popMaximum()
            nextContinuation()
            if len(repository) > 0:
                assert len(repository) == 1
                yield repository[0][1]
                del repository[0]
                
                
    def apply(self,f,x):
        f = f.chase()
        x = x.chase()
        l = LiftedApply(f,x)
        self.incident[f].add(l)
        self.incident[x].add(l)
        return l
    def applyClass(self,f,x): return getOne(self.classes(self._incorporate(self.apply(f,x))))
    def abstract(self,b):
        b = b.chase()
        l = LiftedAbstract(b)
        self.incident[b].add(l)
        return l
    def abstractClass(self,b): return getOne(self.classes(self._incorporate(self.abstract(b))))
    def index(self,i):
        l = LiftedIndex(i)
        self._incorporate(l)
        return l
    def indexClass(self,i): return getOne(self.classes(self._incorporate(self.index(i))))
            

                    
    def incorporate(self,p):
        """Returns a lifted expression"""
        assert isinstance(p,Program)
        
        if p.isPrimitive or p.isInvented:
            l = LiftedLeaf(p)
        elif p.isIndex:
            l = LiftedIndex(p.i)
        elif p.isApplication:
            f = getOne(self.l2e[self.incorporate(p.f)])
            x = getOne(self.l2e[self.incorporate(p.x)])
            l = self.apply(f,x)
        elif p.isAbstraction:
            b = getOne(self.l2e[self.incorporate(p.body)])
            l = self.abstract(b)

        self._incorporate(l)
        return l
    
    def _incorporate(self,l):
        assert isinstance(l,Lifted)

        if l not in self.l2e:
            e = self.newClass(l.freeVariables)
            self.l2e[l] = set()
            self.addEdge(e,l)

        return l

    def addEdge(self, e, l):
        assert isinstance(e, EquivalenceClass)
        assert isinstance(l, Lifted)
        
        self.e2l[e].add(l)
        self.l2e[l].add(e)

    def removeEdge(self, e, l):
        assert isinstance(e, EquivalenceClass)
        assert isinstance(l, Lifted)
        self.e2l[e].remove(l)
        self.l2e[l].remove(e)

    def newClass(self, freeVariables):
        e = EquivalenceClass(self.nextClass, freeVariables)
        self.nextClass += 1
        self.e2l[e] = set()
        self.incident[e] = set()
        return e

    def deleteClass(self,e):
        # Do not delete anything which is still being pointed to
        assert len(self.incident[e]) == 0
        del self.incident[e]
        for l in self.e2l[e]:
            self.l2e[l].remove(e)
        del self.e2l[e]
        if DEBUG:
            # Debugging assertions
            for l,es in self.l2e.items():
                assert e not in es, "the expression %s has %s as an equivalence class, but it is being deleted"%(l,e)
            self.graveyard.append(e)
    def deleteLifted(self,l):
        if l.isAbstraction:
            self.incident[l.body].remove(l)
        elif l.isApplication:
            self.incident[l.f].remove(l)
            if l.x != l.f:
                self.incident[l.x].remove(l)
        
        for e in list(self.l2e[l]):
            self.removeEdge(e,l)
        del self.l2e[l]
        

    def classes(self,l): return self.l2e[l.chase()]
    def getClass(self,l):
        l = l.chase()
        assert len(self.l2e[l]) == 1
        return getOne(self.l2e[l]).chase()
    def parents(self,e): return self.incident[e.chase()]
    def children(self,e): return self.e2l[e.chase()]        
    
    def makeEquivalent(self,a,b, verbose=False):
        """Merges equivalence classes, taking care to update the graph structure"""
        if isinstance(a,Lifted): a = self.getClass(a)
        if isinstance(b,Lifted): b = self.getClass(b)
        a = a.chase()
        b = b.chase()
        
        if a == b: return a

        assert a.freeVariables == b.freeVariables

        z = self.newClass(a.freeVariables)

        parents = self.parents(a) | self.parents(b)
        children = self.children(a) | self.children(b)

        if verbose:
            eprint("Constructed",z,"from the elements of",a,"\t",b)
            eprint("Those elements are",children)

        # Changes all pointers to a/b to instead point to z
        def updateLift(l):
            if l.isAbstraction:
                assert l.body == a or l.body == b
                return self.abstract(z)
            elif l.isApplication:
                f = l.f
                if f == a or f == b: f = z
                x = l.x
                if x == a or x == b: x = z
                assert x != l.x or f != l.f
                return self.apply(f,x)
            else:
                assert False

        # Lifted expression referring to A or B has to be in the incident set
        # Because Z is fresh we are guaranteed that these do not exist yet in the graph
        newParents = {l: updateLift(l) for l in parents}
        for new in newParents.values():
            self.l2e[new] = set()

        if verbose:
            eprint("The parents are:")
            for old, new in newParents.items():
                eprint(old,"|->",new)

        # In case a child is also a parent
        for child in children:
            self.addEdge(z,newParents.get(child,child))

        for old, new in newParents.items():
            es = self.l2e[old]
            for e in es:
                self.addEdge(e, new)
            
        for old in list(newParents.keys()):
            if verbose:
                eprint("Deleting obsolete parent",old)
                self.debug(verbose=False)
            self.deleteLifted(old)

        self.deleteClass(a)
        self.deleteClass(b)
        a.leader = z
        b.leader = z

        # Recursively clean up the graph
        self.mergeDuplicateClasses(set(newParents.values()))
        return z

    def mergeDuplicateClasses(self, ls):
        for l in ls:
            l = l.chase()
            if len(self.classes(l)) > 1:
                es = list(self.classes(l))
                e_ = es[0]
                for e in es[1:]:
                    e_ = self.makeEquivalent(e_,e)
                
        

    def makeEquivalenceClass(self, liftedExpressions):
        return reduce(lambda k1,k2: self.makeEquivalent(k1,k2),
                      { e
                        for l in liftedExpressions
                        for e in self.classes(l) })

    def __len__(self): return len(self.liftedToCanonical)

    def shiftLifted(self, e, d, c=0, visited=None):
        """Shifts a lifted expression, returning a lifted expression"""
        assert isinstance(e,Lifted)
        e = e.chase()
        if d == 0 or all( fv < c for fv in e.freeVariables ): return e
        if visited is None:
            visited = {}
        else:
            old = list(visited.items())
            for (ep,dp,cp),result in old:
                visited[(ep.chase(),dp,cp)] = result.chase()
            if (e,d,c) in visited:
                s = visited[(e,d,c)]
                assert isinstance(s,Lifted)
                return s
        key = (e,d,c)
        
        if e.isApplication:
            e = self.apply(self.shiftEquivalent(e.f,d,c, visited), self.shiftEquivalent(e.x,d,c,visited))
        elif e.isAbstraction:
            e = self.abstract(self.shiftEquivalent(e.body, d, c + 1, visited))
        elif  e.isIndex:
            if e.i >= c: # free
                i = e.i + d
                if i < 0: raise ShiftFailure()
                else: e = LiftedIndex(i)
            else: # bound
                pass
        elif e.isLeaf:
            pass
        else:
            assert False

        self._incorporate(e)
        visited[key] = e
        assert isinstance(e,Lifted)
        return e

    def shiftEquivalent(self, k, d, c=0, visited=None):
        """Shifts equivalence class, returning equivalence class"""
        assert isinstance(k, EquivalenceClass)
        k = k.chase()
        if d == 0 or all( fv < c for fv in k.freeVariables ): return k
        if visited is None:
            visited = {}
        else:
            old = list(visited.items())
            for (ep,dp,cp),result in old:
                visited[(ep.chase(),dp,cp)] = result.chase()

            if (k,d,c) in visited: return visited[(k,d,c)]
        canonical = self.newClass(frozenset(fv + d for fv in k.freeVariables if fv >= c ))
        visited[(k,d,c)] = canonical
        
        return self.makeEquivalent(canonical,
                                   self.makeEquivalenceClass([ self.shiftLifted(l, d, c, visited)
                                                               for l in self.children(k) ]))

    def garbageCollect(self, roots):
        es = { j for r in roots for j in self.reachable(r) }
        ls = { l for e in es for l in self.children(e)  }

        deadClasses = set(self.incident.keys()) - es
        deadExpressions = set(self.l2e.keys()) - ls

        for k in deadClasses:
            del self.e2l[k]
            del self.incident[k]
        for l in deadExpressions:
            del self.l2e[l]

        for e in es:
            for l in deadExpressions:
                if l in self.incident[e]:
                    self.incident[e].remove(l)
                if l in self.e2l[e]:
                    self.e2l[e].remove(l)
        for l in ls:
            for e in deadClasses:
                if e in self.l2e[l]:
                    self.l2e[l].remove(e)
        

    def closedChildren(self,le):
        """Takes as input a lifted expression and generates a sequence of equivalence classes"""
        assert isinstance(le,Lifted)

        visited = set()
        def C(n,l):
            k = self.classes(l)
            assert len(k) == 1
            k = getOne(k)
            
            if k in visited: return
            visited.add(k)
            
            if all( fv - n >= 0 for fv in l.freeVariables ):
                yield from self.classes(self.shiftLifted(l,-n))

            if l.isLeaf or l.isIndex:
                pass
            elif l.isAbstraction:
                for bl in self.children(l.body):
                    yield from C(n + 1, bl)
            elif l.isApplication:
                for z in self.children(l.f) | self.children(l.x):
                    yield from C(n, z)
            else:
                assert False

        yield from C(0,le)

    def shiftEqual(self,n,v,l_):
        """Does |^n v contain l?"""
        if n == 0: return v in self.classes(l_)
        def curse(e,l, c, vVisited, lVisited):
            if e in vVisited or self.getClass(l) in lVisited:
                return False
            vVisited = vVisited|{e}
            lVisited = lVisited|{self.getClass(l)}
            
            if l.isIndex:
                for child in self.children(e):
                    if child.isIndex:
                        if child.i < c: i = child.i
                        else: i = child.i + n
                        if i == l.i: return True
                return False
            elif l.isApplication:
                return any( curse(child.f, lf, c, vVisited, lVisited) and \
                            curse(child.x, lx, c, vVisited, lVisited)
                            for child in self.children(e)
                            if child.isApplication
                            for lf in self.children(l.f)
                            for lx in self.children(l.x) )
            elif l.isAbstraction:
                return any( curse(child.body, lb, c + 1, vVisited, lVisited)
                            for child in self.children(e)
                            if child.isAbstraction
                            for lb in self.children(l.body) )
            elif l.isLeaf:
                return l in self.children(e)

        return curse(v,l_,0,set(),set())
                        
            

    def possibleBodies(self,v,l_):
        """
        v: equivalence class
        le: lifted 
        returns: sequence of equivalence classes
        """
        assert isinstance(v, EquivalenceClass)
        assert isinstance(l_, Lifted)

        allowedExpressions = { child
                               for r in self.reachable(v)|self.reachable(self.getClass(l_))
                               for child in self.children(r) }
            

        """Acyclic is a bit tricky here"""
        """We are recursively constructing a tree, and it is okay at different
        branches of the tree refer to the same class"""
        """But along any particular path from the root to a leaf we should
        only pass through any particular equivalence class once"""
        def S(D, n,le,visited):
            le = le.chase()
            k = self.getClass(le)
            visited = {v.chase() for v in visited }

            if k in visited:
                return set()
            if not any( allowed.chase() == le for allowed in allowedExpressions ):
                return set()

            visited = visited | {k}

            # Optimization: if all of the free variables of v do not
            # occur in le then we can be sure v does not occur in le
            if not ({fv + n for fv in v.freeVariables } <= le.freeVariables):
                shifted = {self.shiftEquivalent(self.getClass(le), 1, n)}
                return shifted

            bodies = set()
            
            # Check if this particular expression is equivalent to the shifted equivalence class
            # Check to see if shifting gives the thing that we are substituting
            # As an optimization we check whether shifting would even give the correct free variables
            if k.freeVariables == frozenset(fv + n for fv in v.freeVariables ):
                # valueExtension = { v_extension.shift(n)
                #                    for v_extension in self.extract(v) }
                # expressionExtension = set(self.extract(k))
                # gt = len(valueExtension&expressionExtension) > 0
                # pr = 
                # assert gt == pr
                if self.shiftEqual(n,v,le):
                    bodies.add(self.indexClass(n))

            if le.isIndex:
                if le.i < n:
                    bodies.add(self.getClass(le))
                elif le.i >= n:
                    bodies.add(self.indexClass(le.i + 1))
            elif le.isAbstraction:
                for b in self.children(le.body):
                    for bp in S(D + 1, n + 1, b, visited):
                        bodies.add(self.abstractClass(bp))
            elif le.isApplication:
                for fp in self.children(le.f):
                    for xp in self.children(le.x):
                        for fpp in S(D + 1,n,fp,visited):
                            for xpp in S(D + 1,n,xp,visited):
                                bodies.add(self.applyClass(fpp,xpp))
            elif le.isLeaf:
                bodies.add(self.getClass(le))
            else:
                assert False

            chasedResult = {b.chase() for b in bodies }
            return chasedResult

        return S(0,0,l_,set())
                

    def inverseBeta1(self,k):
        """1 beta reduction step at the top level
        k: an equivalence class
        returns: a sequence of strongly equivalent classes
        _DOES NOT_ introduce any new equivalences
        """
        k = k.chase()
        # Sequestered immutable part
        l2v = {l.chase(): set(self.closedChildren(l.chase()))
               for l in self.children(k) }
        eprint("Inverse beta",k,"which has",len(l2v),"members",
               "and a total of",sum(len(v) for v in l2v.values() ),"closed children")
        for l,vs in l2v.items():
            eprint("\tProcessing a expression w/ closed children =",len(vs))
            for v in vs:
                l = l.chase()
                v = v.chase()
                    
                for b in set(self.possibleBodies(v,l)):
                    b = self.abstractClass(b)
                    yield self.applyClass(b,v)

    def inverseBetaEverything(self, roots, steps=1):
        if steps <= 0: return 

        with timing("Did one step of inverse beta reduction"):
            roots = {r.chase() for r in roots }
            everything = { j for r in roots for j in self.reachable(r) }
            everything = list(everything)
            eprint("Rewriting",len(everything),"reachable nodes")

            # First perform operations that do not mutate existing equivalence classes
            newEquivalences = {j: {ne for ne in self.inverseBeta1(j)}
                               for j in everything }
            eprint("Calculated",sum(len(eq) for eq in newEquivalences.values() ),"new equivalences")
            for j,eq in newEquivalences.items():
                for e in eq:
                    eprint("Making",j,"equivalent to",e)
                    j = self.makeEquivalent(j,e)
                    #self.debug(verbose=False)

            eprint("Updated equivalences.")

        self.inverseBetaEverything(roots, steps - 1)

    def reachable(self,e):
        """e: an equivalence class"""
        assert isinstance(e, EquivalenceClass)
        visited = set()
        def curse(e):
            if e in visited: return
            visited.add(e)
            for c in self.children(e):
                if c.isAbstraction:
                    curse(c.body)
                elif c.isApplication:
                    curse(c.f)
                    curse(c.x)
        curse(e)
        return visited

    def minimumCost(self,j,given,table=None):
        if table is None: table = {}
        if j in table: return table[j]

        
        if j in given: c = 1
        else:        
            e = self.i2e[j]
            if e.isPrimitive or e.isIndex or e.isInvented: c = 1
            if e.isApplication:
                c = self.minimumCost(e.f,given,table) + self.minimumCost(e.x,given,table)
            if e.isAbstraction:
                c = self.minimumCost(e.body, given, table)
        table[j] = c
        return c

    def bestNewInvention(self,alternatives):
        from collections import Counter
        
        candidates = [ { r for a in a_ for r in self.reachable(a) }
                       for a_ in alternatives.values()                       
        ]
        candidates = Counter(k for ks in candidates for k in ks)
        candidates = {k for k,f in candidates.items() if f >= 2 }
        def scoreInvention(k):
            t = {}
            return sum(min(self.minimumCost(i,{k},t)
                           for i in a )
                       for a in alternatives.values())
        eprint(len(candidates),"candidates to consider for invention")
        j = min(candidates,
                key = scoreInvention)
        return list(self.extract(j))[0]

    def visualize(self, simplify=True, roots=None, pause=False):
        from graphviz import Digraph

        d = Digraph(comment='expression graph')

        if roots is not None:
            reachable = {j for r in roots for j in self.reachable(r)}
            reachable = reachable | \
                        {l for r in reachable for l in self.children(r)  }
        def show(le):
            if roots is None: return True
            return le in reachable

        def nodeCode(l):
            if isinstance(l,EquivalenceClass):
                elements = self.children(l)
                if roots is not None:
                    elements = elements & reachable
                if simplify and len(elements) <= 1: return nodeCode(list(elements)[0])
                return "E%d"%l.name
            if l.isLeaf:
                return str(l.primitive)
            elif l.isIndex:
                return "$%d"%l.i
            elif l.isAbstraction:
                return "abstract%d"%l.body.name
            elif l.isApplication:
                return "apply%d_%d"%(l.f.name,
                                     l.x.name)
            else:
                assert False


        for l in self.l2e.keys():
            if not show(l): continue
            
            code = nodeCode(l)
            if l.isLeaf:
                d.node(code,str(l.primitive))
            elif l.isIndex:
                d.node(code, "$%d"%l.i)
            elif l.isAbstraction:
                d.node(code,
                       "abs")
            elif l.isApplication:
                d.node(code,
                       "@")
            else:
                assert False

        for k in self.e2l.keys():
            if not show(k): continue
            
            elements = self.e2l[k]
            if roots is not None:
                elements = elements & reachable
            if simplify and len(elements) <= 1: continue
            
            code = nodeCode(k)
            d.node(code, "E_%d: {%s}"%(k.name,
                                     ",".join(str(fv) for fv in k.freeVariables)))

            for e in elements:
                d.edge(code, nodeCode(e), color="chocolate")

        for l in self.l2e.keys():
            if not show(l): continue
            
            code = nodeCode(l)
            if l.isAbstraction:
                d.edge(code, nodeCode(l.body))
                assert l in self.incident[l.body]
            elif l.isApplication:
                d.edge(code, nodeCode(l.f), label="f")
                assert l in self.incident[l.f]
                d.edge(code, nodeCode(l.x), label="x")
                assert l in self.incident[l.x]                

        d.render('/tmp/compressionGraph.gv', view=True)
        if pause:
            input("press enter to continue...")

        
                
        
    

def inverseBeta(e,n=1):
    if n > 1:
        for rw in inverseBeta(e,n-1):
            yield from inverseBeta(rw,1)
    else:
        yield from inverseBeta_(e)
        if e.isAbstraction:
            for b in inverseBeta(e.body): yield Abstraction(b)
        elif e.isApplication:
            for f in inverseBeta(e.f): yield Application(f,e.x)
            for x in inverseBeta(e.x): yield Application(e.f,x)
    
def inverseBeta_(e):
    """1 beta reduction step at the top level"""
    for v in set(closedChildren(e)):
        for b in possibleBodies(v,e):
            yield Application(Abstraction(b),v)
            
def closedChildren(e,j=0):
    try:
        yield e.shift(-j)
    except ShiftFailure: pass

    if e.isApplication:
        yield from closedChildren(e.f,j)
        yield from closedChildren(e.x,j)
    elif e.isAbstraction:
        yield from closedChildren(e.body,j+1)

def possibleBodies(v,e,n=0):
    if e == v.shift(n):
        yield Index(n)
    if e.isIndex:
        if e.i < n:
            yield e
        else:
            yield Index(e.i + 1)
    elif e.isAbstraction:
        for b in possibleBodies(v,e.body,n+1):
            yield Abstraction(b)
    elif e.isApplication:
        for f in possibleBodies(v,e.f,n):
            for x in possibleBodies(v,e.x,n):
                yield Application(f,x)
    elif e.isPrimitive or e.isInvented:
        yield e
    else:
        assert False


if __name__ == "__main__":
    from lib.domains.arithmetic.arithmeticPrimitives import *
    from lib.domains.list.listPrimitives import *
    from lib.grammar import *
    bootstrapTarget_extra()
    p1 = Program.parse("(lambda (fold empty $0 (lambda (lambda (cons (- $0) $1)))))")
    p2 = Program.parse("(lambda (fold empty $0 (lambda (lambda (cons (+ $0 1) $1)))))")
    p3 = Program.parse("car")
    p4 = Program.parse("(lambda $0)")

    p = p1
    l = lift(p)
    print()
    n = 2
    i = set()
    with timing("calculated version space for %s"%p):
        for n_ in range(1,n+1):
            i = i | set(l.R(n_).extension())
    gt = set(inverseBeta(p,n))
    print(len(gt - i),"missing from the inversion")
    print(len(i - gt),"missing from ground truth")
    print(len(i),"programs in the extension of the inversion")
    print(len(gt),"programs in the ground truth")
    assert False
    
    for c in i:
        print(c)
        continue
    
        for e in c.extension():
            print(e)
            assert e.betaNormalForm() == getOne(l.extension()).betaNormalForm()
    assert False

    g = ExpressionGraph()
    k1 = g.incorporate(p3)
    g.inverseBetaEverything(g.classes(k1), steps=2)

    g.debug()
    
    # k2 = g.incorporate(p4).body
    # g.makeEquivalent(k1,k2)
    # #g.inverseBetaEverything([k1])
    # g.debug()
    g.visualize()
    for program in g.extract(getOne(g.classes(g.incorporate(p3)))):
        eprint(program)
        eprint(program.betaNormalForm())
    
