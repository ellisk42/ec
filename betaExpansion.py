from program import *

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


def SC(n,e):
    mapping = {True: set()}
    try:
        v = e.shift(-n)
        mapping[v] = {Index(n)}
    except ShiftFailure: pass

    if e.isIndex:
        if e.i < n:
            mapping[True] = {e}
        else:
            mapping[True] = {Index(e.i + 1)}
    elif e.isApplication:
        fm = SC(n,e.f)
        xm = SC(n,e.x)
        for v in fm:
            if v is True: continue
            if not (v in xm): continue
            if not (v in mapping): mapping[v] = set()
            mapping[v].update(Application(fp,xp)
                              for fp in fm[v]
                              for xp in xm[v] )
        for ft in fm.get(True,[]):
            # ft: program
            for v,xt in ((xValue,xBody)
                       for xValue, xBodies in xm.items()
                       for xBody in xBodies):
                if not (v in mapping): mapping[v] = set()
                mapping[v].add(Application(ft,xt))
        for xt in xm.get(True,[]):
            # ft: program
            for v,ft in ((fValue,fBody)
                       for fValue, fBodies in fm.items()
                       for fBody in fBodies):
                if not (v in mapping): mapping[v] = set()
                mapping[v].add(Application(ft,xt))
            
    elif e.isAbstraction:
        bm = SC(n + 1, e.body)
        for v,bodies in bm.items():
            bodies = {Abstraction(b) for b in bodies }
            if v not in mapping:
                mapping[v] = bodies
            else:
                mapping[v].update(bodies)
    elif e.isPrimitive or e.isInvented:
        mapping[True] = mapping.get(True,set())
        mapping[True].add(e)
    else: assert False

    return mapping
        
def recursiveBetaExpand(e,N=1):
    if N > 1:
        for e_ in recursiveBetaExpand(e,N - 1):
            yield e_
            yield from recursiveBetaExpand(e_,N=1)
    else:
        yield from betaExpand(e)
        if e.isApplication:
            for x in recursiveBetaExpand(e.x):
                yield Application(e.f,x)
            for f in recursiveBetaExpand(e.f):
                yield Application(f,e.x)
        if e.isAbstraction:
            for b in recursiveBetaExpand(e.body):
                yield Abstraction(b)
def betaExpand(e):
    for v,bodies in SC(0,e).items():
        if v is True: continue
        for b in bodies:
            yield Application(Abstraction(b),v)


if __name__ == "__main__":
    from arithmeticPrimitives import *
    from listPrimitives import *
    from grammar import *
    bootstrapTarget_extra()
    p1 = Program.parse("(lambda (fold empty $0 (lambda (lambda (cons (- $0 5) $1)))))")
#    p1 = Program.parse("empty")
    p2 = Program.parse("(lambda (fold empty $0 (lambda (lambda (cons (+ $0 $0) $1)))))")

    for e in betaExpand(p1):
        print(e)
        print(e.betaNormalForm())
        print()

    for e in set(inverseBeta_(p1)) - set(betaExpand(p1)):
        print(e)
