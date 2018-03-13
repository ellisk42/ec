from utilities import eprint
from frontier import *
from task import *
from type import *
from program import *
from grammar import *

import gc

def enumerateFrontiers(g, tasks, _=None,
                       solver=None,
                       frontierSize=None,
                       enumerationTimeout=None,
                       CPUs=1,
                       maximumFrontier=None,
                       verbose=True,
                       evaluationTimeout=None):
    '''g: Either a Grammar, or a map from task to grammar.'''
    from time import time

    solvers = {"ocaml": solveForTask_ocaml,
               "pypy": enumerateForTask_pypy,
               "python": enumerateForTask}
    assert solver in solvers, \
        "You must specify a valid solver. options are ocaml, pypy, or python."
    solver = solvers[solver]

    if not isinstance(g, dict): g = {t: g for t in tasks }
    
    start = time()
    frontiers = parallelMap(CPUs,
                            lambda (task, grammar): solver(grammar, task,
                                                           timeout=enumerationTimeout,
                                                           evaluationTimeout = evaluationTimeout,
                                                           maximumFrontier=maximumFrontier),
                            map(lambda t: (t, g[t]), tasks),
                            chunk = 1)
    if verbose:
        eprint("Enumerated %d frontiers in time %f"%(len(g), time() - start))

    times = [t for f,t in frontiers if t is not None]
    frontiers = [f for f,t in frontiers ]
    return frontiers, times

class EnumerationTimeout(Exception): pass

def solveForTask_ocaml(g, task, _ = None, timeout = None, evaluationTimeout = None,
                       maximumFrontier = 10):
    from time import time
    # from multiprocessing import Process, Queue
    from threading import Thread
    from Queue import Queue


    startTime = time()
    
    CPUs = 10

    workers = {}

    lowerBound = 0.
    budgetIncrement = 0.3

    frontier = Frontier([], task)
    q = Queue()

    nextID = 0

    bestSearchTime = None

    while True:
        while len(workers) < CPUs:
            elapsedTime = time() - startTime
            thisTimeout = int(timeout - elapsedTime + 0.5)
            if thisTimeout < 1: break
            
            eprint("Launching worker with timeout",thisTimeout)
            p = Thread(target = _solveForTask_ocaml,
                       args = (nextID, q,
                               g, task, lowerBound, lowerBound+budgetIncrement, budgetIncrement,
                               thisTimeout, evaluationTimeout,
                               maximumFrontier - len(frontier)))
            p.start()
            workers[nextID] = (elapsedTime, p)
            nextID += 1
            lowerBound += budgetIncrement

        ID, newFrontier, searchTime = q.get()
        initialTime, process = workers[ID]

        if searchTime is not None:
            totalTime = initialTime + searchTime
            eprint("(python) Got first solution after %s seconds"%totalTime)
            if bestSearchTime is None: bestSearchTime = totalTime
            else: bestSearchTime = min(totalTime, bestSearchTime)
        frontier = frontier.combine(newFrontier)

        #process.kill()
        del workers[ID]

        if thisTimeout < 1 and len(workers) == 0 and q.empty(): break

    return frontier, bestSearchTime
        
    

def _solveForTask_ocaml(myID, q, g, task, lb, ub, bi,
                        timeout, evaluationTimeout, maximumFrontier):
    import json
    message = {"DSL": {"logVariable": g.logVariable,
                       "productions": [ {"expression": str(p), "logProbability": l}
                                            for l,_,p in g.productions ]},
               "examples": [{"inputs": list(xs), "output": y} for xs,y in task.examples ],
               "programTimeout": evaluationTimeout,
               "solverTimeout": timeout,
               "maximumFrontier": maximumFrontier,
               "name": task.name,
               "lowerBound": lb,
               "upperBound": ub,
               "budgetIncrement": bi,
               "verbose": True}#verbose}
    message = json.dumps(message)
    # with open('message','w') as handle: handle.write(message)
    # eprint(message)
    p = subprocess.Popen(['./solver'],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    response, error = p.communicate(message)
    try:
        response = json.loads(response)
    except:
        eprint("FATAL: Could not load response from ocaml solver.")
        eprint("response:")
        eprint(response)
        eprint("error:")
        eprint(error)
        assert False

    # Remove all entries that do not type correctly
    # This can occur because the solver tries to infer the type
    # Sometimes it infers a type that is too general
    response = [r for r in response if Program.parse(r["program"]).canHaveType(task.request) ]

    frontier = Frontier([FrontierEntry(program = Program.parse(e["program"]),
                                       logLikelihood = e["logLikelihood"],
                                       logPrior = e["logPrior"])
                         for e in response ],
                        task = task)
    #if verbose: eprint(frontier.summarize())

    if frontier.empty: searchTime = None
    else: searchTime = min(e["time"] for e in response)

    q.put((myID, frontier, searchTime))
    

def enumerateForTask_pypy(*arguments, **keywords):
    return callCompiled(enumerateForTask, *arguments, **keywords)

def enumerateForTask(g, task, _ = None,
                     verbose=False,
                     timeout=None,
                     evaluationTimeout=None,
                     frontierSize=None,
                     budgetIncrement=1.0, maximumFrontier = 10**2):
    assert (timeout is not None) or (frontierSize is not None), \
        "enumerateForTask: You must provide either a timeout or a frontier size."
    
    from time import time
    def timeoutCallBack(_1,_2): raise EnumerationTimeout()
    if timeout is not None:
        if verbose: eprint("Alarming timeout for",timeout,"for task",task)
        signal.signal(signal.SIGALRM, timeoutCallBack)
        signal.alarm(timeout)

    timeUntilFirstSolution = None
    frontier = []
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
    except EnumerationTimeout:
        if verbose:
            eprint("Timeout triggered after",time() - starting,"seconds for task",task)
    signal.alarm(0)

    frontier = Frontier(frontier,
                        task = task).topK(maximumFrontier)
    
    return frontier, timeUntilFirstSolution

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
    
