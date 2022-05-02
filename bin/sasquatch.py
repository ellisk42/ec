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

import argparse
parser = argparse.ArgumentParser(description = "")
parser.add_argument("--version", "-v", default=False, action="store_true")
parser.add_argument("--a", "-a", default=1, type=int)
parser.add_argument("--inferior", "-i", default=False, action="store_true")
arguments = parser.parse_args()


UNKNOWN=Program.parse("??")
EXPANSIONS=[Program.parse(pr) for pr in ["fold", "cons", "empty", "+", "1", "0", "2", "?1", "?2"]]+[Index(n) for n in range(4) ] + [Application(UNKNOWN, UNKNOWN), Abstraction(UNKNOWN)]

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
        if isinstance(f, NamedHole):
            eprint("In order to match, we are going to have to write the expression:")
            eprint(program)
            eprint("In terms of the following variables:")
            eprint(xs)
            eprint(f)
            assert False

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
def match_version(template, j):
    global table, saved_matches


    program = table.expressions[j]

    if program.isUnion:
        return {substitution
                for i in program.elements
                for substitution in match_version(template, i)
                if not any(k==table.empty for k in substitution.values() )}
    
    

    if template.isAbstraction:
        matches = set()
        if program.isAbstraction:
            body_matches=match_version(template.body, program.body)
            for bindings in body_matches:
                new_bindings = { k: table.shiftFree(v, -1)
                                 for k,v in bindings.items() }
                if not any(j==table.empty for j in new_bindings.values() ):
                    matches.add(frozendict(new_bindings))
        return matches

    if template.isIndex or template.isPrimitive or template.isInvented:
        if program==template:
            return [frozendict()]
        return []

    if template.isApplication:
        if program.isApplication:
            function_substitutions = match_version(template.f, program.f)
            argument_substitutions = match_version(template.x, program.x)

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
        #j=table.shiftFree(j, -depth)
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
    return optimistic_version(template, corpus) - template.size(named=1, application=.01, abstraction=.01, fragmentVariable=1)

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
    
def optimistic_version(template, corpus, verbose=False):
    global table
    
    total_corpus_size=0
    number_of_matches=0

    inferior_abstraction_checker=None
    
    for j in corpus:
        # original size
        corpus_size = version_size(j)
        
        bindings = match_version(template, j)
        if verbose:
            eprint()
            eprint(bindings)
            for b in bindings:
                for v, binding in b.items():
                    eprint(v, table.extractSmallest(binding, named=None, application=.01, abstraction=.01))

        if len(bindings)>0:
            number_of_matches+=1

            best_binding = min(bindings,
                               key=lambda b: sum(version_size(binding) for v, binding in b.items()))

            # we still pay the cost for each of the arguments
            corpus_size -= sum(version_size(binding)
                               for v, binding in best_binding.items())
            corpus_size -= 1  # we have to put down the symbol for the function
            corpus_size -= .01*len(get_one(bindings)) # and one application for each argument

            if arguments.inferior:
                if inferior_abstraction_checker is None:
                    inferior_abstraction_checker = best_binding
                else:
                    inferior_abstraction_checker = { v: table.intersection(best_binding[v],
                                                                        inferior_abstraction_checker[v])
                                                    for v in best_binding}

            # eprint("corpus_size", corpus_size, "=",version_size(j), "-", 
            #        min( sum(version_size(binding)
            #                             for v, binding in b.items())
            #                        for b in bindings ),
            #        "+ 1 +", .01*len(get_one(bindings)))
        else:
            corpus_size=0
            # eprint("corpus_size", corpus_size, "=",version_size(j))

        total_corpus_size+=corpus_size

    if arguments.inferior and inferior_abstraction_checker is not None:
        if any( v!=table.empty for v in inferior_abstraction_checker.values() ):
            #eprint("inferior", template)
            return NEGATIVEINFINITY

    #eprint("total corpus_size", total_corpus_size)
    if number_of_matches<2: return NEGATIVEINFINITY
    return total_corpus_size

def best_constant_abstraction(corpus):
    possibilities = {e for p in corpus for _, e in  p.walk()}
    return max(possibilities, key=lambda e: utility(e, corpus))

def master_utility(template, corpus):
    if arguments.version:
        global table
        return utility_version(template, [ table.superVersionSpace(table.incorporate(p), arguments.a) for p in corpus ])
    else:
        return utility(template, corpus)

def master_optimistic(template, corpus, verbose=False, coefficient=0.0001):
    if arguments.version:
        global table
        return optimistic_version(template, [ table.superVersionSpace(table.incorporate(p), arguments.a) for p in corpus ], verbose=verbose) - coefficient*template.size(named=1, application=.01, abstraction=.01, fragmentVariable=1)
    else:
        return optimistic(template, corpus)

def master_pessimistic(template, corpus, verbose=False, coefficient=1):
    if not arguments.version: assert False

    
        
    

# check("(lambda (+ ?1 $0))", "(lambda (+ $9 $0))")
# check("(lambda (+ ?1 ?2))", "(lambda (+ $9 $7))")

# check("(fold empty $0 (lambda (lambda (cons (?1 $0 5) $1))))",
#       "(fold empty $0 (lambda (lambda (cons (+ 1 $0) $1))))")

corpus = [Program.parse(program)
          for program in ["(lambda (fold empty $0 (lambda (lambda (cons (+ 1 $0) $1)))))",
                          "(lambda (fold empty $0 (lambda (lambda (cons (* 2 $0) $1)))))",
                          "(lambda (fold empty $0 (lambda (lambda (cons (if (eq? $0 1) 5 7) $1)))))",
                          "(lambda (fold empty $0 (lambda (lambda (cons (* $0 $0) $1)))))"
          ] ]

table=None

def compress(corpus):
    global table
    table = VersionTable(typed=False, #factored=True,
                         identity=False)

    with timing("constructed version spaces"):
        indices = [ table.superVersionSpace(table.incorporate(p), arguments.a) for p in corpus ]

    candidates=["((lambda ??) ??)", "(lambda (?? ??))"]
    for candidate in candidates:
        candidate=Program.parse(candidate)
        eprint("testing ", candidate)
    
        eprint(master_optimistic(candidate, corpus, verbose=True))

    

    starttime = time.time()
    
    best=best_constant_abstraction(corpus)
    best_utility=master_utility(best, corpus)
    eprint("best starting", best, best_utility)
    if best_utility<0:
        best_utility=0
        best=None
    p0=Program.parse("??")
    q=PQ()
    q.push(master_optimistic(p0, corpus), p0)
    #eprint("initial_best_utility", best_utility)
    while len(q):
        p=q.popMaximum()
        
        #eprint(p, master_optimistic(p, corpus))
        r=refinements(p)
        if len(r)==0:
            u = master_utility(p, corpus)
            if u>best_utility:
                eprint(int(time.time()-starttime), "best found so far", p, u)
                best_utility, best = u, p
        else:
            for pp in r:
                if bad_refinement(pp):
                    continue
                u = master_optimistic(pp, corpus)
                if u is None or u<best_utility:
                    continue
                q.push(u, pp)
                



compress(corpus)



