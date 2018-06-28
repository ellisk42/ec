from betaExpansion import *

from program import *

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
    @property
    def isUniverse(self): return False
    @property
    def isUnion(self): return False

    def __repr__(self): return str(self)

    def __or__(self,o):
        if self.isUniverse or o.empty:
            return self
        if o.isUniverse or self.empty: return o
        if self.isUnion and o.isUnion:
            u = self.elements|o.elements
            if len(u) == 1: return next(iter(u))
            return Terms(u)
        if o.isUnion:
            u = {self}|o.elements
            if len(u) == 1: return next(iter(u))
            return Terms(u)
        if self.isUnion:
            u = {o}|self.elements
            if len(u) == 1: return next(iter(u))
            return Terms(u)
        if self == o:
            return self
        return Terms([self,o])

    def __and__(self,o):
        if self.empty or o.empty: return Terms([])
        
        if self.isUniverse: return o
        if o.isUniverse: return self
        
        if self.isUnion:
            u = {e&o for e in self}
            if len(u) == 1: return next(iter(u))
            return Terms(u)
        if o.isUnion:
            u = {e&self for e in o }
            if len(u) == 1: return next(iter(u))
            return Terms(u)
        if o.isApplication and self.isApplication:
            x = self.x&o.x
            f = self.f&o.f
            if x.empty or f.empty: return Terms([])
            return LiftedApply(f,x)
        if o.isAbstraction and self.isAbstraction:
            b = self.body&o.body
            if b.empty: return Terms([])
            return LiftedAbstract(b)
        if (o.isIndex and self.isIndex) or (o.isLeaf and self.isLeaf):
            if o == self: return self
            return Terms([])
        
        return Terms([])

    def _substitutions(self,n):
        s = self.shift(-n)
        if s.empty: return {}
        return {s: LiftedIndex(n)}

    def inversion(self):
        s = {LiftedApply(LiftedAbstract(b),v) for v,b in self.substitutions(0).items()
             if not v.isUniverse}
        if len(s) == 1:
            return next(iter(s))
        return Terms(s)

    def recursiveInversion(self):
        i1 = self.inversion()
        i2 = self._recursiveInversion()
        i = i1 | i2
        return i

    def repeatedExpansion(self,n):
        expansions = [self]
        for _ in range(n):
            expansions.append(expansions[-1].recursiveInversion())
        return Terms(expansions)
            
    

class Universe(Lifted):
    def __init__(self):
        self.freeVariables = frozenset()
    def __eq__(self,o):
        return isinstance(o,Universe)
    def __hash__(self): return 42
    def __str__(self): return "U"

    def extension(self):
        yield Primitive("UNIVERSE",None,None)

    @property
    def empty(self): return False
    @property
    def isUniverse(self): return True



class LiftedLeaf(Lifted):
    def __init__(self, primitive):
        self.primitive = primitive
        self.freeVariables = frozenset()
    def __eq__(self,o): return isinstance(o,LiftedLeaf) and o.primitive == self.primitive
    def __hash__(self): return hash(self.primitive)

    @property
    def isLeaf(self): return True

    def __str__(self): return str(self.primitive)
    def __len__(self): return 1

    @property
    def empty(self): return False

    def shift(self,d,c=0): return self
    def extension(self): yield self.primitive
    
    def substitutions(self,n):
        m = self._substitutions(n)
        m[Universe()] = self
        return m

    def _recursiveInversion(self):
        return Terms([])
                
        

class LiftedApply(Lifted):
    def __init__(self,f,x):
        self.f = f
        self.x = x
        self.freeVariables = self.f.freeVariables | self.x.freeVariables
        
    def __eq__(self,o): return isinstance(o,LiftedApply) and self.f == o.f and self.x == o.x
    def __hash__(self): return hash((hash(self.f),hash(self.x)))

    @property
    def isApplication(self): return True

    def __str__(self): return "(%s %s)"%(str(self.f), str(self.x))
    def __len__(self): return len(self.f)*len(self.x)
    def __contains__(self,p):
        return p.isApplication and p.f in self.f and p.x in self.x
    @property
    def empty(self): return self.f.empty or self.x.empty

    def shift(self,d,c=0):
        f = self.f.shift(d,c)
        x = self.x.shift(d,c)
        if f.empty or x.empty: return Terms([])
        return LiftedApply(f,x)

    def extension(self):
        for f in self.f.extension():
            for x in self.x.extension():
                yield Application(f,x)

    def substitutions(self,n):
        fm = self.f.substitutions(n)
        xm = self.x.substitutions(n)
        mapping = self._substitutions(n)
        for v1,f in fm.items():
            for v2,x in xm.items():
                v = v1&v2
                if v.empty: continue
                a = LiftedApply(f,x)
                if v not in mapping: mapping[v] = a
                else: mapping[v] = mapping[v] | a
        return mapping

    def _recursiveInversion(self):
        return Terms([LiftedApply(self.f.recursiveInversion(), self.x),
                      LiftedApply(self.f, self.x.recursiveInversion())])
        
                                        
    

class LiftedAbstract(Lifted):
    def __init__(self,body):
        self.body = body
        self.freeVariables = frozenset(f - 1 for f in body.freeVariables if f > 0 )
        
    def __eq__(self,o): return isinstance(o,LiftedAbstract) and o.body == self.body
    def __hash__(self): return hash(hash(self.body))

    @property
    def isAbstraction(self): return True

    def __str__(self): return "(lambda %s)"%str(self.body)
    def __len__(self): return len(self.body)
    def __contains__(self,p):
        return p.isAbstraction and p.body in self.body
    @property
    def empty(self):
        return self.body.empty
    

    def shift(self,d,c=0):
        b = self.body.shift(d,c+1)
        if not b.empty:
            return LiftedAbstract(b)
        else:
            return Terms([])
    
    def extension(self):
        for b in self.body.extension():
            yield Abstraction(b)

    def substitutions(self,n):
        m = self._substitutions(n)
        m.update({v: LiftedAbstract(b)
                  for v,b in self.body.substitutions(n + 1).items() })
        return m

    def _recursiveInversion(self):
        return LiftedAbstract(self.body.recursiveInversion())


class LiftedIndex(Lifted):
    def __init__(self,i):
        self.i = i
        self.freeVariables = frozenset([i])
    def __eq__(self,o): return isinstance(o,LiftedIndex) and o.i == self.i
    def __hash__(self): return hash(self.i)
    @property
    def isIndex(self): return True
    @property
    def empty(self): return False

    def __str__(self): return "$%d"%self.i
    def __len__(self): return 1
    def __contains__(self,p):
        return p.isIndex and p.i == self.i

    def shift(self,d,c=0):
        if self.i >= c:
            if self.i + d >= 0:
                return LiftedIndex(self.i + d)
            return Terms([])
        else: return self

    def extension(self): yield Index(self.i)

    def substitutions(self,n):
        m = self._substitutions(n)
        m[Universe()] = self if self.i < n else LiftedIndex(self.i + 1)
        return m

    def _recursiveInversion(self):
        return Terms([])


class Terms(Lifted):
    def __init__(self, elements):
        self.elements = frozenset(e_
                                  for e in elements if not e.empty
                                  for e_ in (e.elements if e.isUnion else [e]) )
        for l in self.elements:
            assert isinstance(l,Lifted)
        self.freeVariables = reduce(lambda x,y: x|y,
                                    (l.freeVariables
                                     for l in self.elements ),
                                    set())
    def __eq__(self,o):
        return isinstance(o,Terms) and self.elements == o.elements
    def __ne__(self,o): return not (self == o)
    def __hash__(self): return hash(self.elements)
    def __str__(self):
        return "{%s}"%(",".join(map(str,list(self.elements))))
    def __contains__(self,p):
        return any( p in t
                    for t in self.elements )
    def __repr__(self): return str(self)
    def __iter__(self): return iter(self.elements)

    @property
    def empty(self): return all( e.empty for e in self  )
    

    def shift(self,d,c=0):
        return Terms({t.shift(d,c) for t in self })

    def substitutions(self,n):
        m = {}
        for e in self:
            for v,b in e.substitutions(n).items():
                if v in m:
                    m[v] = m[v]|b
                else:
                    m[v] = b
        return m

    def extension(self):
        for e in self.elements:        
            yield from e.extension()

    def _recursiveInversion(self):
        return Terms([e.recursiveInversion()
                      for e in self ])


   
            
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
    return t


if __name__ == "__main__":
    from arithmeticPrimitives import *
    from listPrimitives import *
    from grammar import *
    bootstrapTarget_extra()
    p1 = Program.parse("(lambda (fold empty $0 (lambda (lambda (cons (- $0 5) $1)))))")
    p1 = Program.parse("(lambda (fold $0  fold))")
    p2 = Program.parse("(lambda (fold empty $0 (lambda (lambda (cons (+ $0 $0) $1)))))")

    N=3

    with timing("calculated versions base"):
        vs = lift(p1).repeatedExpansion(N)
    with timing("calculated denotation"):
        vs = {e for e in vs.extension()}
    with timing("did the brute force thing"):
        gt = set(recursiveBetaExpand(p1,N=N))
    print(vs - gt)
    print(gt - vs)
    for spurious in vs - gt:
        print("spurious")
        print(spurious)
        print(spurious.betaNormalForm())
        
