from utilities import eprint
from frontier import *
from task import *
from type import *
from program import *
from grammar import *

import gc

# def enumerateFrontiers(g, tasks, _=None,
#                        solver=None,
#                        frontierSize=None,
#                        enumerationTimeout=None,
#                        CPUs=1,
#                        maximumFrontier=None,
#                        verbose=True,
#                        evaluationTimeout=None):
#     '''g: Either a Grammar, or a map from task to grammar.'''
#     from time import time

#     solvers = {"ocaml": solveForTask_ocaml,
#                "pypy": enumerateForTask_pypy,
#                "python": enumerateForTask}
#     assert solver in solvers, \
#         "You must specify a valid solver. options are ocaml, pypy, or python."
#     solver = solvers[solver]

#     if not isinstance(g, dict): g = {t: g for t in tasks }

#     CPUsPerTask = 1 if len(tasks) > CPUs else int(float(CPUs)/len(tasks) + 0.5)
#     eprint("Allocating %d CPUs for each task"%CPUsPerTask)
#     if CPUsPerTask > 1 and solver is not solveForTask_ocaml:
#         eprint("(warning) Using more than one CPU for single task is currently only supported by ocaml.")
    
#     start = time()
#     frontiers = parallelMap(CPUs,
#                             lambda (task, grammar): solver(grammar, task,
#                                                            timeout=enumerationTimeout,
#                                                            CPUs=CPUsPerTask,
#                                                            evaluationTimeout = evaluationTimeout,
#                                                            maximumFrontier=maximumFrontier),
#                             map(lambda t: (t, g[t]), tasks),
#                             chunk = 1)
#     if verbose:
#         eprint("Enumerated %d frontiers in time %f"%(len(g), time() - start))

#     times = [t for f,t in frontiers if t is not None]
#     frontiers = [f for f,t in frontiers ]
#     return frontiers, times

class EnumerationTimeout(Exception): pass

def multithreadedEnumeration(g, tasks, _=None,
                             solver=None,
                             frontierSize=None,
                             enumerationTimeout=None,
                             CPUs=1,
                             maximumFrontier=None,
                             verbose=True,
                             evaluationTimeout=None):
    '''g: Either a Grammar, or a map from task to grammar.'''
    from time import time
    from threading import Thread
    from Queue import Queue

    assert frontierSize is None, "deprecated: frontierSize"

    solvers = {"ocaml": solveForTask_ocaml,
               "pypy": solveForTask_pypy,
               "python": solveForTask_python}
    assert solver in solvers, \
        "You must specify a valid solver. options are ocaml, pypy, or python."
    solver = solvers[solver]

    if not isinstance(g, dict): g = {t: g for t in tasks }
    task2grammar = g

    frontiers = {t: Frontier([], task = t) for t in task2grammar }
    activeTasks = set(task2grammar.keys())
    lowerBounds = {t: 0. for t in task2grammar}
    bestSearchTime = {t: None for t in task2grammar}
    # For each task we keep track of how long we have been working on it
    stopwatches = {t: Stopwatch() for t in tasks }
    totalExplored = 0
    nextID = 0
    # map from ID to task
    workers = {}
    
    def budgetIncrement(lb): return 1.

    startTime = time()

    q = Queue()

    while True:
        activeTasks = {t for t in activeTasks
                       if len(frontiers[t]) < maximumFrontier \
                       and enumerationTimeout - stopwatches[t].elapsed >= 1 }

        finished = len(activeTasks) == 0

        if not finished:
            while len(workers) < CPUs:
                # Sort the tasks by lower bound. Prioritize lower
                # lower bounds to explore shorter programs first
                for t in sorted(activeTasks, key = lambda t: lowerBounds[t])[:CPUs]:
                    thisTimeout = enumerationTimeout - stopwatches[t].elapsed
                    if not stopwatches[t].running: stopwatches[t].start()
                    eprint("Launching [%s] w/ lb = %f, timeout = %f"%(t,lowerBounds[t],thisTimeout))
                    bi = budgetIncrement(lowerBounds[t])
                    p = Thread(target = solver, args = [],
                               kwargs = {"ID": nextID, "q": q, "elapsedTime": stopwatches[t].elapsed,
                                         "g": task2grammar[t], "task": t,
                                         "lowerBound": lowerBounds[t], "upperBound": lowerBounds[t] + bi,
                                         "budgetIncrement": bi, "timeout": thisTimeout,
                                         "evaluationTimeout": evaluationTimeout,
                                         "maximumFrontier": maximumFrontier - len(frontiers[t])})
                    p.start()
                    lowerBounds[t] += bi
                    workers[nextID] = t
                    nextID += 1
                    
        if len(workers) > 0:
            ID, newFrontier, searchTime, explored = q.get()
            task = workers[ID]

            totalExplored += explored
            if totalExplored > 0:
                eprint("(python) Explored %d programs in %s sec. %d programs/sec."%
                       (totalExplored, int(time() - startTime), int(float(totalExplored)/(time() - startTime))))

            if searchTime is not None:
                eprint("(python) Got first solution to %s after %s wall clock seconds"%(task,int(searchTime+0.5)))
                if bestSearchTime[task] is None: bestSearchTime[task] = searchTime
                else: bestSearchTime[task] = min(searchTime, bestSearchTime[task])
            frontiers[task] = frontiers[task].combine(newFrontier)

            # Remove the finished worker and stop it stopwatch if the
            # task is no longer being worked on
            del workers[ID]
            if not any( task == _task for _task in workers.values() ):
                stopwatches[task].stop()

        if finished and len(workers) == 0 and q.empty(): break

    eprint("Completed multithreaded enumeration for",len(tasks),"tasks in",int(time() - startTime),"s")
    pps = float(totalExplored)/(time() - startTime)
    eprint("program evaluations per second:",int(pps))
    eprint("program evaluations per CPU second:",int(pps/CPUs))

    return [frontiers[t] for t in tasks], [bestSearchTime[t] for t in tasks if bestSearchTime[t] is not None ]

                    
            

    

# def solveForTask_ocaml(g, task, _ = None, timeout = None, evaluationTimeout = None,
#                        CPUs = 1,
#                        maximumFrontier = 10):
#     from time import time
#     # from multiprocessing import Process, Queue
#     from threading import Thread
#     from Queue import Queue


#     startTime = time()
    
#     workers = {}

#     lowerBound = 0.
#     budgetIncrement = 1.

#     frontier = Frontier([], task)
#     q = Queue()

#     nextID = 0

#     bestSearchTime = None
#     totalExplored = 0

#     while True:
#         elapsedTime = time() - startTime
#         thisTimeout = int(timeout - elapsedTime + 0.5)
#         programsToFind = maximumFrontier - len(frontier)

#         finished = thisTimeout < 1 or programsToFind <= 0
#         if not finished:
#             while len(workers) < CPUs:
#                 eprint("Launching worker with timeout",thisTimeout)
#                 p = Thread(target = _solveForTask_ocaml,
#                            args = (nextID, q,
#                                    g, task, lowerBound, lowerBound+budgetIncrement, budgetIncrement,
#                                    thisTimeout, evaluationTimeout,
#                                    programsToFind))
#                 p.start()
#                 workers[nextID] = (elapsedTime, p)
#                 nextID += 1
#                 lowerBound += budgetIncrement

#         if len(workers) > 0:
#             # eprint("(python) Blocking on thread queue, have %d in the thread queue..."%len(workers))
#             ID, newFrontier, searchTime, explored = q.get()
#             # eprint("(python) The following worker finished:",ID)
#             initialTime, process = workers[ID]
#             if isinstance(newFrontier, Exception):
#                 exc, isFatal = newFrontier, searchTime
#                 if isFatal: raise exc
#                 del workers[ID]
#                 continue

#             totalExplored += explored
#             if totalExplored > 0:
#                 eprint("(python) Explored %d programs in %s sec. %d programs/sec."%
#                        (totalExplored, int(time() - startTime), int(float(totalExplored)/(time() - startTime))))

#             if searchTime is not None:
#                 totalTime = initialTime + searchTime
#                 eprint("(python) Got first solution after %s seconds"%totalTime)
#                 if bestSearchTime is None: bestSearchTime = totalTime
#                 else: bestSearchTime = min(totalTime, bestSearchTime)
#             frontier = frontier.combine(newFrontier)

#             del workers[ID]

#         if finished and len(workers) == 0 and q.empty(): break

#     return frontier, bestSearchTime
        
    

def solveForTask_ocaml(_ = None,
                       ID = None, q = None,
                       elapsedTime = 0.,
                       g = None, task = None,
                       lowerBound = None, upperBound = None, budgetIncrement = None,
                       timeout = None,
                       evaluationTimeout = None, maximumFrontier = None):
    import json
    message = {"DSL": {"logVariable": g.logVariable,
                       "productions": [ {"expression": str(p), "logProbability": l}
                                            for l,_,p in g.productions ]},
               "examples": [{"inputs": list(xs), "output": y} for xs,y in task.examples ],
               "programTimeout": evaluationTimeout,
               "solverTimeout": int(timeout + 0.5),
               "maximumFrontier": maximumFrontier,
               "name": task.name,
               "lowerBound": lowerBound,
               "upperBound": upperBound,
               "budgetIncrement": budgetIncrement,
               "verbose": True}#verbose}
    message = json.dumps(message)
    # with open('message','w') as handle: handle.write(message)
    # eprint(message)
    try:
        p = subprocess.Popen(['./solver'],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            response, error = p.communicate(message)
            response = json.loads(response)
        except Exception as exc:
            exc = ValueError("Could not load response from ocaml solver: ", exc)
            q.put((ID, exc, None, None))
            raise exc
    except OSError as exc:
        q.put((ID, exc, True, None))
        raise exc

    pc = response[u"programCount"]
    # Remove all entries that do not type correctly
    # This can occur because the solver tries to infer the type
    # Sometimes it infers a type that is too general
    response = [r for r in response[u"solutions"] if Program.parse(r["program"]).canHaveType(task.request) ]
    
    frontier = Frontier([FrontierEntry(program = p,
                                       logLikelihood = e["logLikelihood"],
                                       logPrior = g.closedLogLikelihood(task.request, p))
                         for e in response
                         for p in [Program.parse(e["program"])] ],
                        task = task)

    if frontier.empty: searchTime = None
    else: searchTime = min(e["time"] for e in response) + elapsedTime

    q.put((ID, frontier, searchTime, pc))

def solveForTask_pypy(_ = None,
                      ID = None, q = None,
                      elapsedTime = 0.,
                      g = None, task = None,
                      lowerBound = None, upperBound = None, budgetIncrement = None,
                      timeout = None,
                      evaluationTimeout = None, maximumFrontier = None):
    """Executes inside of the thread; puts its results into the queue q"""
    frontier, T, pc = callCompiled(enumerateForTask,
                                   g,task,
                                   timeout = timeout,
                                   evaluationTimeout = evaluationTimeout,
                                   maximumFrontier = maximumFrontier,
                                   budgetIncrement = budgetIncrement,
                                   lowerBound = lowerBound, upperBound = upperBound)
    q.put((ID, frontier, T, pc))

def solveForTask_python(_ = None,
                        ID = None, q = None,
                        elapsedTime = 0.,
                        g = None, task = None,
                        lowerBound = None, upperBound = None, budgetIncrement = None,
                        timeout = None,
                        evaluationTimeout = None, maximumFrontier = None):
    """Executes inside of the thread; puts its results into the queue q"""
    frontier, T, pc = callFork(enumerateForTask,
                               g,task,
                               timeout = timeout,
                               evaluationTimeout = evaluationTimeout,
                               maximumFrontier = maximumFrontier,
                               budgetIncrement = budgetIncrement,
                               lowerBound = lowerBound, upperBound = upperBound)
    q.put((ID, frontier, T, pc))

def enumerateForTask_pypy(*arguments, **keywords):
    return callCompiled(enumerateForTask, *arguments, **keywords)

def enumerateForTask(g, task, _ = None,
                     verbose=False,
                     timeout=None,
                     evaluationTimeout=None,
                     frontierSize=None,
                     lowerBound = 0.,
                     upperBound = 100.,
                     budgetIncrement=1.0, maximumFrontier = 10**2):
    assert (timeout is not None) or (frontierSize is not None), \
        "enumerateForTask: You must provide either a timeout or a frontier size."
    
    from time import time
    def timeoutCallBack(_1,_2): raise EnumerationTimeout()
    if timeout is not None:
        if verbose: eprint("Alarming timeout for",timeout,"for task",task)
        signal.signal(signal.SIGALRM, timeoutCallBack)
        signal.alarm(int(timeout+0.5))

    timeUntilFirstSolution = None
    frontier = []
    starting = time()
    previousBudget = lowerBound
    budget = lowerBound + budgetIncrement
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
                if valid(likelihood):
                    if verbose:
                        eprint("Hit",task.name,"with the program",p,"which has prior",prior,"after",time() - starting,"seconds")
                    if frontier == []: timeUntilFirstSolution = time() - starting                        
                    frontier.append(FrontierEntry(program = p,
                                                  logPrior = prior,
                                                  logLikelihood = likelihood))

                # If the alarm is triggered during evaluation,
                # it will be caught by the catchall exception handler
                # And so we have to time ourselves out
                if timeout is not None and time() - starting > timeout:
                    signal.alarm(0)
                    raise EnumerationTimeout
            if verbose:
                eprint("Enumerated %d programs of satisfying:"%(numberOfPrograms),
                       "%d < MDL <= %d."%(int(previousBudget),int(budget)))
            
            previousBudget = budget
            budget += budgetIncrement
            totalNumberOfPrograms += numberOfPrograms
            if verbose:
                eprint("\tTotal elapsed time: %d seconds. Total number of programs evaluated: %d. Task: %s."% \
                       (time() - starting, totalNumberOfPrograms, task))
            if frontierSize is not None and totalNumberOfPrograms > frontierSize: break
            if budget > upperBound: break
    except EnumerationTimeout:
        if verbose:
            eprint("Timeout triggered after",time() - starting,"seconds for task",task)
    signal.alarm(0)

    frontier = Frontier(frontier,
                        task = task).topK(maximumFrontier)
    
    return frontier, timeUntilFirstSolution, numberOfPrograms

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
    
