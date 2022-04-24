try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module
import frozendict
from dreamcoder.program import *
from dreamcoder.vs import *
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

def bad_refinement(template):
    previous=None
    for _, t in template.walk():
        if t.isNamedHole:
            if previous is None:
                if str(t)!="?1": return True
            else:
                if int(str(t)[1:]) not in [previous,previous+1]:
                    return True
            previous=int(str(t)[1:])
    return False

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

@memoize
def match_version(template, j, depth=0):
    global table, saved_matches


    program = table.expressions[j]

    if program.isUnion:
        return {substitution
                for i in program.elements
                for substitution in match_version(template, i, depth)
                if not any(k==table.empty for k in substitution.values() )}
    
    

    if template.isAbstraction:
        if program.isAbstraction:
            return match_version(template.body, program.body, depth+1)
        return set()

    if template.isIndex or template.isPrimitive or template.isInvented:
        if program==template:
            return [frozendict()]
        return []

    if template.isApplication:
        if program.isApplication:
            function_substitutions = match_version(template.f, program.f, depth)
            argument_substitutions = match_version(template.x, program.x, depth)

            possibilities=[]
            for fs in function_substitutions:
                for xs in argument_substitutions:
                    new_substitution={}
                    domain = set(fs.keys())|set(xs.keys())
                    failure=False
                    for v in domain:
                        overlap=table.intersection(fs.get(v, table.universe),
                                                   xs.get(v, table.universe))
                        if overlap==table.empty:
                            failure=True
                            break
                        new_substitution[v]=overlap
                    if not failure:
                        possibilities.append(frozendict(new_substitution))
            return set(possibilities)
        else:
            return []

    if isinstance(template, NamedHole):
        j=table.shiftFree(j, -depth)
        if j==table.empty:
            return []
        return [frozendict({template.name:j})]

    if isinstance(template, FragmentVariable):
        return [frozendict()]

    assert False

def utility(template, corpus):
    corpus_size=0
    k=usages(template)
    
    for p in corpus:
        try:
            bindings = match(template, p)
            assert None not in bindings

            # original size
            corpus_size += p.size(named=1, application=.01, abstraction=.01)

            # we still pay the cost for each of the arguments
            corpus_size -= sum(binding.size(application=.01, abstraction=.01)
                               for v, binding in bindings.items() )
            corpus_size -= 1  # we have to put down the symbol for the function
            corpus_size -= .01*len(bindings) # and one application for each argument
            
        except MatchFailure:
            ...
            # rewriting fails so we just add and subtract the same thing
    return corpus_size - template.size(named=1, application=.01, abstraction=.01)

def utility_version(template, corpus):
    basic = optimistic_version(template, corpus)
    if basic==None: return NEGATIVEINFINITY
    return basic - template.size(named=1, application=.01, abstraction=.01)

def optimistic(template, corpus):
    corpus_size=0
    k=usages(template)
    
    for p in corpus:
        try:
            bindings = match(template, p)

            # original size
            corpus_size += p.size(named=1, application=.01, abstraction=.01)

            # we still pay the cost for each of the arguments
            corpus_size -= sum(binding.size(application=.01, abstraction=.01)
                               for v, binding in bindings.items() if v!=None)
            corpus_size -= 1  # we have to put down the symbol for the function
            corpus_size -= .01*len(bindings) # and one application for each argument
            
        except MatchFailure:
            ...
            # rewriting fails so we just add and subtract the same thing
    return corpus_size #- template.size({"named":1, "application": .01, "abstraction": .01})

@memoize
def version_size(j):
    global table
    return table.extractSmallest(j, named=None, application=.01, abstraction=.01).size(named=None, application=.01, abstraction=.01)

def get_one(s):
    for x in s: return x
    assert False
    
def optimistic_version(template, corpus):
    global table
    
    corpus_size=0
    terrible=True
    for j in corpus:
        # original size
        corpus_size += version_size(j)
        
        bindings = match_version(template, j)
        # eprint()
        # eprint()
        # for b in bindings:
        #     for v, binding in b.items():
        #         eprint(v, table.extractSmallest(binding, named=None, application=.01, abstraction=.01))

        if len(bindings)>0:
            terrible=False
            

            # we still pay the cost for each of the arguments
            corpus_size -= min( sum(version_size(binding)
                                    for v, binding in b.items())
                               for b in bindings )
            corpus_size -= 1  # we have to put down the symbol for the function
            corpus_size -= .01*len(get_one(bindings)) # and one application for each argument
    if terrible: return NEGATIVEINFINITY
    return corpus_size

def best_constant_abstraction(corpus):
    possibilities = {e for p in corpus for _, e in  p.walk()}
    return max(possibilities, key=lambda e: utility(e, corpus))
    

check("(lambda (+ ?1 $0))", "(lambda (+ $9 $0))")
check("(lambda (+ ?1 ?2))", "(lambda (+ $9 $7))")

check("(fold empty $0 (lambda (lambda (cons (?1 $0 5) $1))))",
      "(fold empty $0 (lambda (lambda (cons (+ 1 $0) $1))))")

corpus = [Program.parse(program)
          for program in ["(fold empty $0 (lambda (lambda (cons (+ $0 2) $1))))",
                          "(fold empty $0 (lambda (lambda (cons (* 2 $0) $1))))",
                          "(fold empty $0 (lambda (lambda (cons (if (eq? $0 1) 5 7) $1))))",
                          "(fold empty $0 (lambda (lambda (cons (* $0 $0) $1))))"
          ] ]

table=None

def compress(corpus, a=0):
    global table
    table = VersionTable(typed=False, factored=True)

    with timing("constructed version spaces"):
        indices = [ table.superVersionSpace(table.incorporate(p), a) for p in corpus ]

    eprint("testing target")
    target=Program.parse("(fold empty $0 (lambda (lambda (cons (?1 $0) $1))))")
    eprint(target,utility_version(target, indices))
    #assert False
    
    best=best_constant_abstraction(corpus)
    best_utility=utility(best, corpus)
    p0=Program.parse("??")
    q=PQ()
    q.push(optimistic_version(p0, indices), p0)
    while len(q):
        p=q.popMaximum()
        #eprint(p, utility_version(p, indices))
        r=refinements(p)
        if len(r)==0:
            u = utility_version(p, indices)
            if u>best_utility:
                eprint("best found so far", p, u)
                best_utility, best = u, p
        else:
            for pp in r:
                if bad_refinement(pp):
                    continue
                u = optimistic_version(pp, indices)
                if u is None or u<best_utility:
                    continue
                q.push(u, pp)



compress(corpus, 1)

