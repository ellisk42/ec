import frozendict
from dreamcoder.program import *
from dreamcoder.vs import *
from dreamcoder.grammar import *
from dreamcoder.fragmentUtilities import primitiveSize
from colorama import Fore

UNKNOWN=None

@memoize(1)
def version_size(table, j):
    expression = table.extractSmallest(j, named=None, application=.01, abstraction=.01)
    if expression is None:
        return POSITIVEINFINITY
    return expression.size(named=None, application=.01, abstraction=.01)

@memoize(1,2)
def match_version_comprehensive(table, template, program, rewriting_steps):
    """
    template: program
    program: program
    returns: set(tuple(float, dict(fv->index)))
    (c, {fv->j}) means that a subexpression of size c can be replaced with the template by mapping fv to anything in the version space j
    """
    if not hasattr(program, "super_index"):
        program.super_index = table.superVersionSpace(table.incorporate(program), rewriting_steps)
    
    j = program.super_index
    
    root = match_version(table, template, j)
    if len(root)>0:
        my_cost = program.size(named=None, application=.01, abstraction=.01)
        return {(my_cost, substitution)
                for substitution in root}

    if program.isAbstraction:
        return match_version_comprehensive(table, template, program.body, rewriting_steps)
    

    if program.isIndex or program.isPrimitive or program.isInvented:
        return []

    if program.isApplication:
        return {(subcost, substitution)
                for i in [program.f, program.x]
                for subcost, substitution in match_version_comprehensive(table, template, i, rewriting_steps)
                if not any(k==table.empty for k in substitution.values() )}

    assert False
    
    

@memoize(1,2)
def match_version(table, template, j):
    """
    template: program
    j: version space index
    returns: set(dict(fv->index))
    """

    program = table.expressions[j]

    if program.isUnion:
        return {substitution
                for i in program.elements
                for substitution in match_version(table, template, i)
                if not any(k==table.empty for k in substitution.values() )}
    
    

    if template.isAbstraction:
        matches = set()
        if program.isAbstraction:
            body_matches=match_version(table, template.body, program.body)
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
            function_substitutions = match_version(table, template.f, program.f)
            argument_substitutions = match_version(table, template.x, program.x)

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


def utility(table, template, corpus, rewriting_steps, verbose=False, inferior=False, coefficient=0.01):
    global UNKNOWN
    
    total_corpus_size=0
    number_of_matches=0

    inferior_abstraction_checker=None

    for frontier in corpus:
        # how big is the smallest program which solves the task, without doing any compression?
        baseline_task_size = min(version_size(table, table.incorporate(program))
                                 for program in frontier)

        # list of (binding, utility) pairs
        # utility: baseline_task_size - task_size_with_program_rewritten_with_this_invention
        bindings_utilities = [(None, 0.)]
        
        for program in frontier:
            
            if template is UNKNOWN or template.parent is UNKNOWN or id(program) in template.parent.matches:
                bindings = match_version_comprehensive(table, template, program, rewriting_steps)
            else:
                bindings = []
            
            if verbose:
                eprint()
                eprint(bindings)
                for subcost, b in bindings:
                    for v, binding in b.items():
                        eprint(subcost, v, table.extractSmallest(binding, named=None, application=.01, abstraction=.01))

            if len(bindings)>0:
                if template is not UNKNOWN: template.matches.add(id(program))

                bindings = list(bindings)

                # we have to put down the symbol for the function,
                # and one application for each argument,
                # hence  + 1 + .01*len(b)
                size_of_original_program = version_size(table, table.incorporate(program))
                for subexpression_cost, b in bindings:
                    size_of_invention_application = sum(version_size(table, binding) for v, binding in b.items()) + 1 + .01*len(b)
                    size_of_thing_invention_is_replacing = subexpression_cost
                    size_of_rewritten_program = size_of_original_program - size_of_thing_invention_is_replacing + size_of_invention_application
                    bindings_utilities.append((b, baseline_task_size - size_of_rewritten_program))
            

        
        best_binding, corpus_size_delta = max(bindings_utilities,
                                              key=lambda bu: bu[1])
        if best_binding is None:
            continue
        

        total_corpus_size+=corpus_size_delta
        number_of_matches+=1

        if inferior:
            if inferior_abstraction_checker is None:
                inferior_abstraction_checker = best_binding
            else:
                inferior_abstraction_checker = { v: table.intersection(best_binding[v],
                                                                       inferior_abstraction_checker[v])
                                                 for v in best_binding}
        
    if inferior and inferior_abstraction_checker is not None:
        if any( v!=table.empty for v in inferior_abstraction_checker.values() ):
            #eprint("inferior", template)
            return NEGATIVEINFINITY

    #eprint("total corpus_size", total_corpus_size)
    if number_of_matches<2: return NEGATIVEINFINITY
    return total_corpus_size - coefficient*template.size(named=1, application=.01, abstraction=.01, fragmentVariable=1)


def refinements(template, expansions):
    if template.isAbstraction:
        return [Abstraction(b) for b in refinements(template.body, expansions)]
    
    if template.isApplication:
        f = refinements(template.f, expansions)
        if f:
            return [Application(ff, template.x) for ff in f]
        x = refinements(template.x, expansions)
        return [Application(template.f, xx) for xx in x]

    if template.isPrimitive or template.isIndex or template.isInvented or template.isNamedHole:
        return []

    if isinstance(template, FragmentVariable):
        return expansions

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

def rewrite_with_invention(table, template, invention, rewriting_steps, program):
    """
    Returns the version space of all ways that the program could be rewritten with the invention
    """
    if not hasattr(program, "super_index"):
        program.super_index = table.superVersionSpace(table.incorporate(program), rewriting_steps)
    j = program.super_index
    
    root = match_version(table, template, j)

    possibilities = [j]

    def substitution2application(s):
        a = table.incorporate(invention)
        for i in reversed(range(len(s))):
            a = table.apply(a, s[f"?{i+1}"])
        return a
    
    possibilities.extend([ substitution2application(substitution)
                           for substitution in root])
    
    if program.isAbstraction:
        possibilities.append(table.abstract(rewrite_with_invention(table, template, invention, rewriting_steps, program.body)))

    if program.isIndex or program.isPrimitive or program.isInvented:
        possibilities.append(table.incorporate(program))

    if program.isApplication:
        possibilities.append(table.apply(
            rewrite_with_invention(table, template, invention, rewriting_steps, program.f),
            rewrite_with_invention(table, template, invention, rewriting_steps, program.x)))
    
    return table.union(possibilities)

def sasquatch_grammar_induction(g0, frontiers,
                                _=None,
                                pseudoCounts=1.,
                                a=3,
                                rewriting_steps=1,
                                aic=1.,
                                topK=20,
                                topI=50,
                                structurePenalty=1.,
                                inferior=False,
                                rerank=True, 
                                CPUs=1):
    global UNKNOWN
    arity = a
    
    originalFrontiers = frontiers
    frontiers = [frontier for frontier in frontiers if not frontier.empty]
    eprint("Inducing a grammar from", len(frontiers), "frontiers")

    def objective(g, fs):
        ll = sum(g.frontierMarginal(f) for f in fs )
        sp = structurePenalty * sum(primitiveSize(p) for p in g.primitives)
        return ll - sp - aic*len(g.productions)

    def rewrite(program, request, template, invention):
        vs = rewrite_with_invention(table, template, invention, rewriting_steps, program)
        smallest = table.extractSmallest(vs, named=None, application=.01, abstraction=.01)
        try:
            return EtaLongVisitor(request).execute(smallest)
        except EtaExpandFailure:
            return program
            

    def rewrite_and_score(grammar, frontiers, template):
        # ?1 |-> $0
        # ?2 |-> $1
        # ?3 |-> $2
        # etc
        invention = template
        for ii in range(arity):
            invention = invention.substitute(Program.parse(f"?{ii+1}"), Index(ii))
        invention = Invented(invention.wrap_in_abstractions(invention.numberOfFreeVariables))

        rewritten_frontiers = [ Frontier([ FrontierEntry(rewrite(fe.program, frontier.task.request,
                                                                 template, invention),
                                                         logPrior=0.,
                                                         logLikelihood=fe.logLikelihood)
                                           for fe in frontier],
                                         task=frontier.task)
                                for frontier in frontiers ]
        g = Grammar.uniform([invention] + grammar.primitives,
                            continuationType=grammar.continuationType).\
            insideOutside(rewritten_frontiers,
                          pseudoCounts=pseudoCounts)
        
        rewritten_frontiers = [g.rescoreFrontier(f) for f in rewritten_frontiers]

        newScore = objective(g, rewritten_frontiers)

        return g, invention, rewritten_frontiers, newScore

        

    with timing("Estimated initial grammar production probabilities"):
        g0 = g0.insideOutside(frontiers, pseudoCounts)
    oldScore = objective(g0, frontiers)
    eprint("Starting grammar induction score",oldScore)

    while True:
        table = VersionTable(typed=False, identity=False)
        with timing("constructed %d-step version spaces"%rewriting_steps):
            for f in frontiers:
                for e in f:
                    table.superVersionSpace(table.incorporate(e.program), rewriting_steps)
        eprint("Enumerated %d distinct version spaces"%len(table.expressions))

        corpus = [ [e.program for e in f] for f in frontiers  ]

        UNKNOWN=Program.parse("??")
        UNKNOWN.parent=None

        basic_primitives = []
        EXPANSIONS=[Program.parse(f"?{ii+1}") for ii in range(arity)]
        EXPANSIONS.extend([ existing_primitive
                            for _1,_2,existing_primitive in g0.productions ])
        EXPANSIONS.extend([Index(n) for n in range(5) ])
        EXPANSIONS.extend([Application(UNKNOWN, UNKNOWN), Abstraction(UNKNOWN)])

        starttime = time.time()
        best_utility=0
        best_score=oldScore
        best=None
        p0=UNKNOWN
        q=PQ()
        q.push(utility(table, p0, corpus, rewriting_steps, inferior=inferior), p0)

        # p1=Program.parse("(lambda (fix1 $0 (lambda (lambda (if (empty? $0) ?1 (?2 (car $0) ($1 (cdr $0))))))))")
        # q.push(999999999991, p1)

        pops=0
        utility_calculations=1
        while len(q):
            p=q.popMaximum()
            pops+=1

            #eprint(p, master_optimistic(p, corpus))
            r=refinements(p, EXPANSIONS)
            if len(r)==0:
                u = utility(table, p, corpus, rewriting_steps, inferior=False, coefficient=1)
                newScore = rewrite_and_score(g0, frontiers, p)[-1]
                
                if (rerank and newScore>best_score) or \
                   ((not rerank) and u>best_utility): 
                    color = Fore.GREEN
                    best = p
                else:
                    color = Fore.YELLOW if u>0 else Fore.RED
                eprint(color, int(time.time()-starttime), "sec", pops, "pops", utility_calculations, "utility calculations", "best found so far", p, "utility", u, "probabilistic score", newScore, '\033[0m')

                best_utility = max(u, best_utility)
                best_score = max(newScore, best_score)
            else:
                for pp in r:
                    if bad_refinement(pp):
                        continue
                    pp.parent=p
                    pp.matches=set()
                    u = utility(table, pp, corpus, rewriting_steps, inferior=inferior,
                                coefficient=1)
                    utility_calculations+=1
                    if u is None or u<best_utility:
                        continue
                    q.push(u, pp)

        if best is None:
            eprint("No invention looks promising so we are done")
            break

        g, invention, rewritten_frontiers, newScore = rewrite_and_score(g0, frontiers, best)

        version_size.clear()
        match_version_comprehensive.clear()
        match_version.clear()

        if newScore > oldScore:
            eprint("Score can be improved to", newScore)
            g0, frontiers, oldScore = g, rewritten_frontiers, newScore
            eprint("New library function occurs in",
                   sum(str(invention) in str(frontier.bestPosterior.program)
                       for frontier in frontiers ),
                   "MAP solutions:")
            for frontier in frontiers:
                if str(invention) in str(frontier.bestPosterior.program):
                    eprint(frontier.task.name, "\t", frontier.bestPosterior.program)
            eprint()
        else:
            eprint("Score does not actually improve, finishing")
            break
            
            
        
        
        
