from utilities import eprint
from frontier import *
from task import *
from type import *
from program import *
from grammar import *

import gc

def enumerateFrontiers(g, frontierSize, tasks, CPUs=1,
                       maximumFrontier=None,
                       verbose=True,
                       evaluationTimeout=None):
    '''g: Either a Grammar, or a map from task to grammar.'''
    from time import time

    if isinstance(g, dict):
        start = time()
        f = parallelMap(CPUs,            
                        lambda (task, grammar): enumerateFrontiers(grammar, frontierSize, [task],
                                                                   CPUs=1, verbose=False,
                                                                   maximumFrontier=maximumFrontier)[0],
                        map(lambda t: (t, g[t]), tasks))
        if verbose:
            eprint("Enumerated %d frontiers in time %f"%(len(g), time() - start))
        return f

    start = time()
    uniqueRequests = list({ t.request for t in tasks })
    frontiers = dict(parallelMap(
        CPUs,
        lambda request: (request, iterativeDeepeningEnumeration(g, request, frontierSize,
                                                                showDescriptionLength = verbose)),
        uniqueRequests))
    # for _,f in frontiers.iteritems():
    #     for _,e in f: eprint(e)
    # for t in tasks: eprint(t)
    totalNumberOfPrograms = sum(len(f) for f in frontiers.values())
    totalNumberOfFrontiers = len(frontiers)

    frontiers = {t: frontiers[t.request] for t in tasks}

    if verbose:
        eprint("Enumerated %d frontiers with %d total programs in time %fsec" %
               (totalNumberOfFrontiers, totalNumberOfPrograms, time() - start))

    # I think that this is no longer needed...
    # # In general these programs have considerable overlap, reusing
    # # many subtrees. This code will force identical trees to only be
    # # represented by a single object on the heap.
    # if isinstance(g, dict):
    #     with timing("Coalesced trees"):
    #         share = ShareVisitor()
    #         frontiers = {t: [ (l, share.execute(p)) for l,p in f ]
    #                      for t,f in frontiers.iteritems() }
    #         share = None # collect it
    #         gc.collect()

    start = time()
    # We split up the likelihood calculation and the frontier construction
    # This is so we do not have to serialize and deserialize a bunch of programs
    # programLikelihoods: [ {indexInfrontiers[task]: likelihood} (for each task)]
    programLikelihoods = parallelMap(CPUs, lambda task:
                                     {j: logLikelihood
                                      for j, (_, program) in enumerate(frontiers[task])
                                      for logLikelihood in [
                                          task.logLikelihood(program, timeout=evaluationTimeout)
                                      ]
                                      if valid(logLikelihood)},
                                     tasks)

    frontiers = constructFrontiers(frontiers, programLikelihoods, tasks, maximumFrontier)

    dt = time() - start
    if verbose:
        eprint("Scored frontiers in time %fsec (%f/program)" % (dt, dt / totalNumberOfPrograms))

    return frontiers

class EnumerationTimeout(Exception): pass

def enumerateForTask(g, task, timeout = 5, budgetIncrement=1.0, maximumFrontier = 10**4):
    from time import time
    def timeoutCallBack(_1,_2): raise EnumerationTimeout()
    signal.signal(signal.SIGALRM, timeoutCallBack)
    signal.alarm(timeout)
    
    frontier = {}
    starting = time()
    previousBudget = 0.
    budget = previousBudget + budgetIncrement
    try:
        totalNumberOfPrograms = 0
        while len(frontier) < maximumFrontier:
            numberOfPrograms = 0
            for prior,_,p in enumeration(g, Context.EMPTY, [], task.request, 
                                         maximumDepth = 99,
                                         upperBound = budget,
                                         lowerBound = previousBudget):
                if len(frontier) >= maximumFrontier: break
                #eprint(p, -prior)
                descriptionLength = -prior
                # Shouldn't see it on this iteration
                if descriptionLength > budget: assert False
                # Should already have seen it
                if descriptionLength <= previousBudget:
                    assert False, \
                        "Enumerated program %s with MDL %f. But the lower bound was %f, and the upper bound was %f."%(p,descriptionLength,previousBudget,budget)

                numberOfPrograms += 1
                
                likelihood = task.logLikelihood(p)
                if valid(likelihood):
                    eprint("Hit",task.name,"with the program",p,"which has prior",prior,"after",time() - starting,"seconds")
                    frontier[p] = (prior, likelihood)
            eprint("Enumerated %d programs of satisfying:"%(numberOfPrograms),
                   "%d < MDL <= %d."%(int(previousBudget),int(budget)))
            
            previousBudget = budget
            budget += budgetIncrement
            totalNumberOfPrograms += numberOfPrograms
            eprint("\tTotal elapsed time: %d seconds. Total number of programs evaluated: %d."% \
                   (time() - starting, totalNumberOfPrograms))
    except EnumerationTimeout: pass
    signal.alarm(0)        

    return Frontier([FrontierEntry(program = p,
                                   logLikelihood = likelihood,
                                   logPrior = prior)
                     for p,(likelihood, prior) in frontier.iteritems() ],
                    task = task)

def iterativeDeepeningEnumeration(g, request, frontierSize, budget=2.0, budgetIncrement=1.0, showDescriptionLength = False):
    """Returns a list of (log likelihood, program)"""
    frontier = []
    while len(frontier) < frontierSize:
        frontier = [(l, p) for l, _, p in enumeration(g, Context.EMPTY, [], request, budget)]
        budget += budgetIncrement
    if showDescriptionLength: eprint("Enumerated up to %f nats"%(budget - budgetIncrement))
    # This will trim the frontier to be exactly frontierSize We do
    # this for small frontier sizes; the idea is that if the frontier
    # is small then you probably want exactly that many programs
    if frontierSize <= 2000: return sorted(frontier, key=lambda (l, p): l, reverse=True)[:frontierSize]
    return frontier


def enumeration(g, context, environment, request, upperBound, maximumDepth = 20, lowerBound = 0.):
    '''Enumerates all programs whose MDL satisfies: lowerBound < MDL <= upperBound'''
    if upperBound <= 0 or maximumDepth == 1: return 

    if request.isArrow():
        v = request.arguments[0]
        for l, newContext, b in enumeration(g, context, [v] + environment,
                                            request.arguments[1],
                                            upperBound = upperBound,
                                            lowerBound = lowerBound,
                                            maximumDepth = maximumDepth):
            yield l, newContext, Abstraction(b)

    else:
        candidates = g.buildCandidates(request, context, environment,
                                       normalize = True)
        
        for l, t, p, newContext in candidates:
            mdl = -l
            if not (mdl <= upperBound): continue
            
            xs = t.functionArguments()
            # eprint("Enumerating arguments for function",p,"which has been requesting types",xs)
            for aL,aK,application in\
                enumerateApplication(g, newContext, environment, p, xs,
                                     upperBound = upperBound + l,
                                     lowerBound = lowerBound + l,
                                     maximumDepth = maximumDepth - 1):
                yield aL+l, aK, application


def enumerateApplication(g, context, environment,
                         function, argumentRequests,
                         # Upper bound on the description length of all of the arguments
                         upperBound,
                         # Lower bound on the description length of all of the arguments
                         lowerBound = 0.,
                         maximumDepth = 20):
    if upperBound <= 0 or maximumDepth == 1: return 

    if argumentRequests == []:
        # eprint("Enumerating application of %s with no arguments."%(function))
        # eprint("\tL",lowerBound)
        # eprint("\tU",upperBound)
        if lowerBound < 0. and 0. <= upperBound:
            yield 0., context, function
        else: return 
    else:
        argRequest = argumentRequests[0].apply(context)
        laterRequests = argumentRequests[1:]
        for argL, newContext, arg in enumeration(g, context, environment, argRequest,
                                                 upperBound = upperBound,
                                                 lowerBound = 0.,
                                                 maximumDepth = maximumDepth):
            newFunction = Application(function, arg)
            for resultL, resultK, result in enumerateApplication(g, newContext, environment, newFunction,
                                                                 laterRequests,
                                                                 upperBound = upperBound + argL,
                                                                 lowerBound = lowerBound + argL,
                                                                 maximumDepth = maximumDepth):
                yield resultL + argL, resultK, result


def constructFrontiers(frontiers, programLikelihoods, tasks, maxFrontier):
    newFrontiers = []
    for programLikelihood, task in zip(programLikelihoods, tasks):
        entries = []
        for j, (logPrior, program) in enumerate(frontiers[task]):
            ll = programLikelihood.get(j, NEGATIVEINFINITY)
            entry = FrontierEntry(program, logPrior=logPrior, logLikelihood=ll)
            entries.append(entry)
        frontier = Frontier(entries, task=task).removeZeroLikelihood()
        newFrontiers.append(frontier.topK(maxFrontier))
    return newFrontiers

def solveSingleTask(grammar, task, maximumBudget = 15):
    if isinstance(task, DifferentiableTask):
        rememberOld = True
        history = set([])
    else: rememberOld = False
    for budget in range(2, maximumBudget):
        for _,_,p in enumeration(grammar, Context.EMPTY, [], task.request, budget):
            if rememberOld:
                if p in history: continue
                history.add(p)
            l = task.logLikelihood(p)
            if valid(l): return l,p
    return None

def benchmarkSynthesisTimes(result, tasks, _ = None, timeout = None, CPUs = None):
    if result.parameters['useRecognitionModel']:
        assert hasattr(result, 'recognitionModel') and result.recognitionModel is not None, \
            "Checkpoint was trained using a recognition model but it does not have a saved recognition model."

    times = parallelMap(CPUs, lambda task: benchmarkSynthesisTime(result, task, timeout), tasks)
    timeouts = sum(t == None for t in times)
    successes = sum(t != None for t in times)
    if successes > 0:
        average = sum(t[0] for t in times if t != None)/float(successes)
        deviation = (sum( (t[0] - average)**2 for t in times if t != None )/float(successes))**0.5
        standardError = deviation/(float(successes)**0.5)
    eprint("BENCHMARK:")
    eprint("Solves %d/%d = %d%%"%(successes, len(tasks), int(100.*successes/len(tasks))))
    if successes > 0:
        eprint("Synthesis time %f +/- %f sec"%(average, standardError))
        average = sum(t[1] for t in times if t != None)/float(successes)
        deviation = (sum( (t[1] - average)**2 for t in times if t != None )/float(successes))**0.5
        standardError = deviation/(float(successes)**0.5)
        eprint("Expected log P[t|p] =",average,"+/-",standardError)

def benchmarkSynthesisTime(result, task, timeout):
    grammar = result.grammars[-1]
    
    from time import time
    import signal
    
    startTime = time()
    if result.parameters['useRecognitionModel']:
        # Because grammar induction is the last step of EC, the
        # recognition model is actually trained for the second to last
        # grammar
        grammar = result.grammars[-2]        
        features = result.recognitionModel.featureExtractor.featuresOfTask(task)
        variables, productions = result.recognitionModel(features)
        grammar = Grammar(variables.data[0],
                          [ (productions.data[k],t,p)
                            for k,(_,t,p) in enumerate(grammar.productions) ])

    try:
        elapsed = time() - startTime
        solution = callCompiled(solveSingleTask,
                                grammar, task, maximumBudget = 99999,
                                compiledTimeout = timeout - elapsed)
        dt = time() - startTime
        if dt > timeout: raise CompiledTimeout()
        l,p = solution
        eprint("Solved",task,"w/",p,"(log likelihood of task given program:",l,").","in time",dt)
        return dt,l
    except CompiledTimeout:
        eprint("Failed to solve",task,"in time",timeout)
        return None
