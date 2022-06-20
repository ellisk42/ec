try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module
import frozendict
from dreamcoder.program import *
from dreamcoder.vs import *
from dreamcoder.domains.list.listPrimitives import *
from colorama import Fore

basic_primitives = list(set(McCarthyPrimitives()+bootstrapTarget_extra()+basePrimitives()+[Program.parse(str(ii)) for ii in range(10) ]))

class MatchFailure(Exception): pass

import argparse
parser = argparse.ArgumentParser(description = "")
parser.add_argument("--version", "-v", default=False, action="store_true")
parser.add_argument("--a", "-a", default=1, type=int)
parser.add_argument("--old", default=False, action="store_true")
parser.add_argument("--inferior", "-i", default=False, action="store_true")
parser.add_argument("--thorough", "-t", default=False, action="store_true")
arguments = parser.parse_args()


UNKNOWN=Program.parse("??")
UNKNOWN.parent=None
EXPANSIONS=[Program.parse(pr) for pr in ["?1", "?2", "?3"]]+basic_primitives+[Index(n) for n in range(4) ] + [Application(UNKNOWN, UNKNOWN), Abstraction(UNKNOWN)]

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
    for surroundingAbstractions, t in template.walk():
        if t.isNamedHole:
            if previous is None:
                if str(t)!="?1": return True
            else:
                if int(str(t)[1:]) not in [previous,previous+1]:
                    return True
            previous=int(str(t)[1:])
        if t.isIndex:
            if t.i >= surroundingAbstractions:
                return True
        if t.isApplication:
            if t.f.isAbstraction:
                return True
    return False

def usages(template):
    count={}
    for _, e in template.walk():
        if e.isNamedHole:
            count[e.name]=count.get(e.name, 0)+1
    return count
            



"""version space approach"""

@memoize()
def match_version_comprehensive(template, program):
    """
    template: program
    program: program
    returns: set(tuple(float, dict(fv->index)))
    (c, {fv->j}) means that a subexpression of size c can be replaced with the template by mapping fv to anything in the version space j
    """
    global table

    if not hasattr(program, "super_index"):
        program.super_index = table.superVersionSpace(table.incorporate(program), arguments.a)
    
    j = program.super_index
    
    root = match_version(template, j)
    if len(root)>0 or not arguments.thorough:
        my_cost = program.size(named=None, application=.01, abstraction=.01)
        return {(my_cost, substitution)
                for substitution in root}

    if program.isAbstraction:
        return match_version_comprehensive(template, program.body)

    if program.isIndex or program.isPrimitive or program.isInvented:
        return []

    if program.isApplication:
        return {(subcost, substitution)
                for i in [program.f, program.x]
                for subcost, substitution in match_version_comprehensive(template, i)
                if not any(k==table.empty for k in substitution.values() )}

    assert False
    
    

@memoize()
def match_version(template, j):
    """
    template: program
    j: version space index
    returns: set(dict(fv->index))
    """
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
                #for k,v in bindings.items(): eprint("shifting", table.extractSmallest(v), table.extractSmallest(table.shiftFree(v, 1)))
                new_bindings = { k: table.shiftFree(v, 1)
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

def utility_version(template, corpus):
    return optimistic_version(template, corpus) - template.size(named=1, application=.01, abstraction=.01, fragmentVariable=1)

@memoize()
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

    for program in corpus:
        if template is UNKNOWN or template.parent is UNKNOWN or id(program) in template.parent.matches:
            bindings = match_version_comprehensive(template, program)
        else:
            bindings = []
            
        if verbose:
            eprint()
            eprint(bindings)
            for subcost, b in bindings:
                for v, binding in b.items():
                    eprint(subcost, v, table.extractSmallest(binding, named=None, application=.01, abstraction=.01))

        if len(bindings)>0:
            number_of_matches+=1
            if template is not UNKNOWN: template.matches.add(id(program))

            bindings = list(bindings)

            # we have to put down the symbol for the function,
            # and one application for each argument,
            # hence  + 1 + .01*len(b)
            bindings_utilities = [ subexpression_cost - \
                                   (sum(version_size(binding) for v, binding in b.items()) + 1 + .01*len(b))
                                   for subexpression_cost, b in bindings ]
            best_binding, corpus_size_delta = max(zip(bindings, bindings_utilities),
                                                   key=lambda bu: bu[1])
            best_binding = best_binding[1]  # strip off the program size

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
            corpus_size_delta=0
            # eprint("corpus_size", corpus_size, "=",version_size(j))

        total_corpus_size+=corpus_size_delta

    if arguments.inferior and inferior_abstraction_checker is not None:
        if any( v!=table.empty for v in inferior_abstraction_checker.values() ):
            #eprint("inferior", template)
            return NEGATIVEINFINITY

    #eprint("total corpus_size", total_corpus_size)
    if number_of_matches<2: return NEGATIVEINFINITY
    return total_corpus_size



"""basic approach"""

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
    return corpus_size 


def best_constant_abstraction(corpus):
    possibilities = {e for p in corpus for _, e in  p.walk() if not bad_refinement(e)}
    return max(possibilities, key=lambda e: master_utility(e, corpus))

def master_utility(template, corpus):
    if arguments.version:
        return utility_version(template, corpus)
    else:
        return utility(template, corpus)

def master_optimistic(template, corpus, verbose=False, coefficient=0.01):
    if arguments.version:
        return optimistic_version(template, corpus, verbose=verbose) - coefficient*template.size(named=1, application=.01, abstraction=.01, fragmentVariable=1)
    else:
        return optimistic(template, corpus)

def master_pessimistic(template, corpus, verbose=False, coefficient=1):
    if not arguments.version: assert False

    
        
    

# check("(lambda (+ ?1 $0))", "(lambda (+ $9 $0))")
# check("(lambda (+ ?1 ?2))", "(lambda (+ $9 $7))")

# check("(fold empty $0 (lambda (lambda (cons (?1 $0 5) $1))))",
#       "(fold empty $0 (lambda (lambda (cons (+ 1 $0) $1))))")

easy_corpus = [Program.parse(program)
          for program in ["(lambda (fold $0 empty (lambda (lambda (cons (+ 1 $1) $0)))))",
                          "(lambda (fold $0 empty (lambda (lambda (cons (* $1 2) $0)))))",
                          "(lambda (fold $0 empty (lambda (lambda (cons (if (eq? $1 1) 5 7) $0)))))",
                          "(lambda (fold $0 empty (lambda (lambda (cons (* $1 $1) $0)))))"
          ] ]
fold_corpus = [Program.parse(program)
          for program in ["(lambda (fix1 $0 (lambda (lambda (if (empty? $0) empty (cons (car $0) (cons (car $0) ($1 (cdr $0)))))))))",
                          "(lambda (fix1 $0 (lambda (lambda (if (empty? $0) 0 (+ (car $0) ($1 (cdr $0))))))))",
                          "(lambda (fix1 $0 (lambda (lambda (if (empty? $0) 1 (- (car $0) ($1 (cdr $0))))))))",
                          "(lambda (fix1 $0 (lambda (lambda (if (empty? $0) (cons 0 empty) (cons (car $0) ($1 (cdr $0))))))))",
                          "(lambda (fix1 $0 (lambda (lambda (if (empty? $0) (empty? empty) (if (car $0) ($1 (cdr $0)) (eq? 1 0)))))))",
                          
          ] ]
unfold_corpus = [Program.parse(program)
          for program in ["(lambda (fix1 $0 (lambda (lambda (if (eq? 0 $0) empty (cons (- 0 $0) ($1 (+ 1 $0))))))))",
                          "(lambda (fix1 $0 (lambda (lambda (if (empty? (cdr $0)) empty (cons (car $0) ($1 (cdr $0))))))))", 
                          "(lambda (fix1 $0 (lambda (lambda (if (eq? $0 0) empty (cons (+ $0 1) ($1 (- $0 1))))))))",
                          "(lambda (fix1 $0 (lambda (lambda (if (eq? $0 0) empty (cons (- 0 $0) ($1 (+ $0 1))))))))"
          ] ]
corpus = unfold_corpus

table=None

def compress(corpus):
    global table
    table = VersionTable(typed=False, #factored=True,
                         identity=False)

    with timing("constructed version spaces"):
        indices = [ table.superVersionSpace(table.incorporate(p), arguments.a) for p in corpus ]

    candidates=[#"((lambda ??) ??)", "(lambda (?? ??))",
                #"(lambda (fold $0 empty (lambda (lambda (cons (?1 $1) $0)))))"
        ]
    for candidate in candidates:
        candidate=Program.parse(candidate)
        eprint("testing ", candidate)

        candidate.parent = UNKNOWN
    
        eprint(master_optimistic(candidate, corpus, verbose=True))

    
    
    starttime = time.time()
    
    # best=best_constant_abstraction(corpus)
    # best.parent = UNKNOWN
    # best_utility=master_utility(best, corpus)
    # eprint("best starting", best, best_utility)
    # if best_utility<0:
    best_utility=0
    best=None
    p0=UNKNOWN
    q=PQ()
    q.push(master_optimistic(p0, corpus), p0)
    #eprint("initial_best_utility", best_utility)
    pops=0
    utility_calculations=1
    while len(q):
        p=q.popMaximum()
        pops+=1
        
        #eprint(p, master_optimistic(p, corpus))
        r=refinements(p)
        if len(r)==0:
            u = master_utility(p, corpus)
            if u>best_utility:
                eprint(Fore.GREEN, int(time.time()-starttime), "sec", pops, "pops", utility_calculations, "utility calculations", "best found so far", p, u, '\033[0m')
                best_utility, best = u, p
        else:
            for pp in r:
                if bad_refinement(pp):
                    continue
                pp.parent=p
                pp.matches=set()
                u = master_optimistic(pp, corpus)
                utility_calculations+=1
                if u is None or u<best_utility:
                    continue
                q.push(u, pp)
                


if arguments.old:
    induceGrammar_Beta(Grammar.uniform(basic_primitives),
                   [Frontier.dummy(p, tp=None) for p in corpus],
                   CPUs=1,
                   a=arguments.a,
                   structurePenalty=0.5)
else:
    # eprint("running the old stuff")
    # compress(corpus)
    from dreamcoder.sasquatch import sasquatch_grammar_induction
    g0 = Grammar.uniform(basic_primitives)
    sasquatch_grammar_induction(g0, [Frontier.dummy(p, tp=p.infer().makeDummyMonomorphic())
                                     for p in corpus],
                                a=3,
                                inferior=arguments.inferior)



