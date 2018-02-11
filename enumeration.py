from utilities import eprint
from frontier import *
from task import *
from type import *
from program import *
from grammar import *

import gc

def enumerateFrontiers(g, tasks, _=None,
                       frontierSize=None,
                       enumerationTimeout=None,
                       CPUs=1,
                       maximumFrontier=None,
                       verbose=True,
                       evaluationTimeout=None):
    '''g: Either a Grammar, or a map from task to grammar.'''
    from time import time

    if not isinstance(g, dict): g = {t: g for t in tasks }
    
    start = time()
    frontiers = parallelMap(CPUs,            
                            lambda (task, grammar): enumerateForTask(grammar, task,
                                                                     frontierSize=frontierSize,
                                                                     timeout=enumerationTimeout,
                                                                     evaluationTimeout = evaluationTimeout,
                                                                     verbose=False,
                                                                     maximumFrontier=maximumFrontier),
                            map(lambda t: (t, g[t]), tasks))
    if verbose:
        eprint("Enumerated %d frontiers in time %f"%(len(g), time() - start))
    return f

class EnumerationTimeout(Exception): pass

def enumerateForTask(g, task, _ = None,
                     verbose=False,
                     timeout=None,
                     evaluationTimeout=None,
                     frontierSize=None,
                     budgetIncrement=1.0, maximumFrontier = 10**2):
    verbose = True
    assert (timeout is not None) or (frontierSize is not None), \
        "enumerateForTask: You must provide either a timeout or a frontier size."
    
    from time import time
    def timeoutCallBack(_1,_2): raise EnumerationTimeout()
    if timeout is not None:
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
                descriptionLength = -prior
                # Shouldn't see it on this iteration
                assert descriptionLength <= budget
                # Should already have seen it
                assert descriptionLength > previousBudget

                numberOfPrograms += 1
                
                likelihood = task.logLikelihood(p, timeout=evaluationTimeout)
                if verbose and valid(likelihood):
                    eprint("Hit",task.name,"with the program",p,"which has prior",prior,"after",time() - starting,"seconds")
                    frontier[p] = (prior, likelihood)
            if verbose:
                eprint("Enumerated %d programs of satisfying:"%(numberOfPrograms),
                       "%d < MDL <= %d."%(int(previousBudget),int(budget)))
            
            previousBudget = budget
            budget += budgetIncrement
            totalNumberOfPrograms += numberOfPrograms
            if verbose:
                eprint("\tTotal elapsed time: %d seconds. Total number of programs evaluated: %d."% \
                       (time() - starting, totalNumberOfPrograms))
            if frontierSize is not None and totalNumberOfPrograms > frontierSize: break
    except EnumerationTimeout: pass
    if timeout is not None:
        signal.alarm(0) 

    frontier = Frontier([FrontierEntry(program = p,
                                       logLikelihood = likelihood,
                                       logPrior = prior)
                         for p,(likelihood, prior) in frontier.iteritems() ],
                        task = task)
    frontier = frontier.topK(maximumFrontier)
    return frontier

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

    elapsed = time() - startTime
    frontier = callCompiled(enumerateForTask,
                            grammar, task,
                            maximumFrontier = 1,
                            timeout = timeout - elapsed)
    dt = time() - startTime
    if dt > timeout or len(frontier) == 0: return None
    l = solution.entries[0].logLikelihood
    p = solution.entries[0].program
    eprint("Solved",task,"w/",p,"(log likelihood of task given program:",l,").","in time",dt)
    return dt,l
    
