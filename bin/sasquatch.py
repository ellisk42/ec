try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.program import *
from dreamcoder.domains.list.listPrimitives import *
basePrimitives()
bootstrapTarget_extra()

class MatchFailure(Exception): pass

UNKNOWN=Program.parse("??")
EXPANSIONS=[Index(n) for n in range(4) ] + [Application(UNKNOWN, UNKNOWN), Abstraction(UNKNOWN)] + [Program.parse(pr) for pr in ["fold", "cons", "empty", "+", "1", "0", "2", "?1", "?2"]]

def refinements(template):
    if template.isAbstraction:
        return [Abstraction(b) for b in refinements(template.body)]
    
    if template.isApplication:
        f = refinements(template.f)
        if f:
            return [Application(ff, template.x) for ff in f]
        x = refinements(template.x)
        return [Application(template.f, xx) for xx in x]

    if template.isPrimitive or template.isIndex or template.isInvented or template.isNamedHole:
        return []

    if isinstance(template, FragmentVariable):
        return EXPANSIONS

def usages(template):
    count={}
    for _, e in template.walk():
        if e.isNamedHole:
            count[e.name]=count.get(e.name, 0)+1
    return count
            

def match(template, program, depth=0, bindings=None):
    if bindings is None:
        bindings = {}

    if template.isAbstraction:
        if program.isAbstraction:
            return match(template.body, program.body, depth+1, bindings)
        raise MatchFailure()

    if template.isIndex:
        if program.isIndex and program==template:
            return bindings
        raise MatchFailure()

    if template.isPrimitive:
        if program.isPrimitive and program==template:
            return bindings
        raise MatchFailure()

    if template.isInvented:
        if program==template:
            return bindings
        raise MatchFailure()

    if template.isApplication:
        f, xs = template.applicationParse()
        if False and isinstance(f, NamedHole):
            eprint("In order to match, we are going to have to write the expression:")
            eprint(program)
            eprint("In terms of the following variables:")
            eprint(xs)

            body = program.shift(len(xs))
            for n, x in enumerate(reversed(xs)):
                body = body.substitute(x.shift(len(xs)), Index(n))
            program = body.wrap_in_abstractions(len(xs))
            
            try: program=program.shift(-depth)
            except ShiftFailure: raise MatchFailure()
            
            if f.name in bindings and program!=bindings[f.name]:
                raise MatchFailure()
            bindings[f.name]=program
            return bindings
            
        if program.isApplication:
            match(template.f, program.f, depth, bindings)
            return match(template.x, program.x, depth, bindings)
        else:
            raise MatchFailure()

    if isinstance(template, NamedHole):
        try:
            program=program.shift(-depth)
        except ShiftFailure:
            raise MatchFailure()
        if template.name in bindings and program!=bindings[template.name]:
            raise MatchFailure()
        bindings[template.name]=program
        return bindings

    if isinstance(template, FragmentVariable):
        bindings[None]=bindings.get(None, [])+[program]
        return bindings

    assert False

def check(template, program):
    program = Program.parse(program)
    template = Program.parse(template)

    eprint()
    eprint("original program", program)
    eprint("template", template)

    try:
        bindings = match(template, program)
    except MatchFailure:
        eprint("no match")
        return 

    eprint(bindings)
    if None in bindings: del bindings[None]

    template = template.shift(len(bindings))
    for i, wildcard in enumerate(sorted(bindings.keys())):
        template = template.substitute(NamedHole(wildcard), Index(i))
    template = template.wrap_in_abstractions(len(bindings))
    eprint(template)

    application = template

    for i, wildcard in enumerate(sorted(bindings.keys(), reverse=True)):
        application = Application(application, bindings[wildcard])

    eprint(application)
    eprint(application.betaNormalForm())

def utility(template, corpus):
    corpus_size=0
    k=usages(template)
    
    for p in corpus:
        try:
            bindings = match(template, p)
            assert None not in bindings
            corpus_size += template.size()
            # savings from repeat occurrences of variables
            corpus_size += sum(binding.size()*(k[v]-1) for v, binding in bindings.items() )
            corpus_size -= 1 # we have to put down the symbol for the function
        except MatchFailure:
            ...
            #corpus_size+=p.size()
    return corpus_size - template.size()

def optimistic(template, corpus):
    corpus_size=0
    k=usages(template)

    terrible=True
    for p in corpus:
        try:
            bindings = match(template, p)
            corpus_size += sum(binding.size()*(k[v]-1) for v, binding in bindings.items() if v!=None)
            corpus_size += sum(h.size() for h in bindings.get(None, []))
            terrible=False
        except MatchFailure:
            ...
    if terrible:
        return None
    return corpus_size

def best_constant_abstraction(corpus):
    possibilities = {e for p in corpus for _, e in  p.walk()}
    return max(possibilities, key=lambda e: utility(e, corpus))
    

check("(lambda (+ ?1 $0))", "(lambda (+ $9 $0))")
check("(lambda (+ ?1 ?2))", "(lambda (+ $9 $7))")

check("(fold empty $0 (lambda (lambda (cons (?1 $0 5) $1))))",
      "(fold empty $0 (lambda (lambda (cons (+ 1 $0) $1))))")

corpus = [Program.parse(program)
          for program in ["(fold empty $0 (lambda (lambda (cons (+ 2 $0) $1))))",
                          "(fold empty $0 (lambda (lambda (cons (* 2 $0) $1))))",
                          "(fold empty $0 (lambda (lambda (cons (car $0) $1))))"] ]

best=best_constant_abstraction(corpus)
best_utility=utility(best, corpus)
p0=Program.parse("??")
q=PQ()
q.push(optimistic(p0, corpus), p0)
while len(q):
    p=q.popMaximum()
    r=refinements(p)
    if len(r)==0:
        u = utility(p, corpus)
        if u>best_utility:
            eprint(p, u)
            best_utility, best = u, p
    else:
        for pp in r:
            u = optimistic(pp, corpus)
            if u is None or u<=best_utility:
                continue
            q.push(u, pp)
    


        
