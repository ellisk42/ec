from utilities import eprint
from frontier import *
from task import *
from type import *
from program import *
from grammar import *
from pregex import pregex

import gc
import traceback
import subprocess
import threading


# Initialise with the command you'll want to run, eg c = Command("./solver")
# Then c.run(msg, timeout) gives msg to c's stdin, let it run for timeout
# seconds, then ask nicely to stop. For compatibility with the previous code,
# this returns (r, e) as return by communicate itself.
class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None
        self.r = None
        self.e = None

    def run(self, msg, timeout):
        def target():
            self.process = subprocess.Popen(self.cmd,
                                            stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE)
            self.r, self.e = self.process.communicate(bytes(msg, encoding="utf-8"))

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)   # Wait for finish in less than timeout
                               # If not ready yet, then:
                               # Ask him nicely to stop, then wait again
        if thread.is_alive() and self.process is not None:
            try:
                self.process.send_signal(signal.SIGUSR1)
            except AttributeError:
                eprint("A process was 'None'. Investiagate")
            thread.join()

        return (self.r,self.e)

command = Command("./solver")

def multithreadedEnumeration(g, tasks, likelihoodModel, _=None,
                             solver=None,
                             frontierSize=None,
                             enumerationTimeout=None,
                             CPUs=1,
                             maximumFrontier=None,
                             verbose=False,
                             evaluationTimeout=None):
    '''g: Either a Grammar, or a map from task to grammar.'''
    from time import time

    # We don't use actual threads but instead use the multiprocessing
    # library. This is because we need to be able to kill workers.
    from multiprocessing import Process, Queue

    assert frontierSize is None, "deprecated: frontierSize"

    solvers = {"ocaml": solveForTask_ocaml,
               "pypy": solveForTask_pypy,
               "python": solveForTask_python}
    assert solver in solvers, \
        "You must specify a valid solver. options are ocaml, pypy, or python."
    solver = solvers[solver]

    if not isinstance(g, dict): g = {t: g for t in tasks }
    task2grammar = g

    frontiers = {t: Frontier([], task=t) for t in task2grammar }

    # Tasks which have not yet been solved
    activeTasks = set(task2grammar.keys())

    # Largest lower bound of any workerthat is assigned to a task
    lowerBounds = {t: 0. for t in task2grammar}

    # Map from task to the shortest time to find a program solving it
    bestSearchTime = {t: None for t in task2grammar}

    # For each task we keep track of how long we have been working on it
    stopwatches = {t: Stopwatch() for t in tasks }

    # Total number of evaluated programs
    totalExplored = 0

    # Each worker is assigned a unique ID number
    nextID = 0

    # map from ID to task
    workers = {}

    def numberOfHits(f):
        return sum( e.logLikelihood == 0. for e in f)

    def budgetIncrement(lb):
        # Very heuristic - not sure what to do here
        if lb < 24.:
            return 1.
        elif lb < 27.:
            return 0.5
        else:
            return 0.25

    startTime = time()

    # Workers put their messages in here
    q = Queue()

    while True:
        activeTasks = {t for t in activeTasks
                       if len(frontiers[t]) < maximumFrontier \
                       and stopwatches[t].elapsed <= enumerationTimeout }

        finished = len(activeTasks) == 0

        if not finished:
            while len(workers) < CPUs:
                # Sort the tasks by lower bound. Prioritize lower
                # lower bounds to explore shorter programs first
                for t in sorted(activeTasks, key=lambda t: lowerBounds[t])[:CPUs-len(workers)]:
                    thisTimeout = enumerationTimeout - stopwatches[t].elapsed
                    if not stopwatches[t].running: stopwatches[t].start()
                    eprint("Launching [%s] w/ lb = %f, timeout = %f"%(t,lowerBounds[t],thisTimeout))
                    bi = budgetIncrement(lowerBounds[t])
                    launchParallelProcess(wrapInThread(solver),
                                          q=q, ID=nextID,
                                          elapsedTime=stopwatches[t].elapsed,
                                          g=task2grammar[t],
                                          task=t,
                                          lowerBound=lowerBounds[t],
                                          upperBound=lowerBounds[t] + bi,
                                          budgetIncrement=bi,
                                          timeout=thisTimeout,
                                          likelihoodModel=likelihoodModel,
                                          evaluationTimeout=evaluationTimeout,
                                          maximumFrontier=maximumFrontier - numberOfHits(frontiers[t]))
                    lowerBounds[t] += bi
                    workers[nextID] = t
                    nextID += 1

        if len(workers) > 0:
            message = Bunch(q.get())
            ID = message.ID
            if message.result == "fork":
                assert False, "Forking message is deprecated"
            elif message.result == "failure":
                eprint("PANIC! Exception in child worker:", message.exception)
                eprint(message.stacktrace)
                assert False
            elif message.result == "success":
                frontier, searchTime, explored = message.value
                task = workers[ID]

                totalExplored += explored
                if totalExplored > 0:
                    eprint("(python) Explored %d programs in %s sec. %d programs/sec. CPU load: %s."%
                           (totalExplored,
                            int(time() - startTime),
                            int(float(totalExplored)/(time() - startTime)),
                            CPULoad()))

                if searchTime is not None:
                    if bestSearchTime[task] is None:
                        eprint("(python) Got first solution to %s after %s wall clock seconds"%(task,int(searchTime+0.5)))
                        bestSearchTime[task] = searchTime
                    else: bestSearchTime[task] = min(searchTime, bestSearchTime[task])
                frontiers[task] = frontiers[task].combine(frontier)

                # Remove the finished worker
                del workers[ID]

                # stop it stopwatch if the task is no longer being
                # worked on
                if not any( task == _task for _task in workers.values() ):
                    stopwatches[task].stop()

        if finished and len(workers) == 0 and q.empty(): break

    eprint("Completed multithreaded enumeration for",len(tasks),"tasks in",int(time() - startTime),"s")
    pps = float(totalExplored)/(time() - startTime)
    eprint("program evaluations per second:",int(pps))
    eprint("program evaluations per CPU second:",int(pps/CPUs))

    return [frontiers[t] for t in tasks], [bestSearchTime[t] for t in tasks if bestSearchTime[t] is not None ]






def wrapInThread(f):
    """
    Returns a function that is designed to be run in a thread/threadlike process.
    Result will be either put into the q
    """

    def _f(*a,**k):
        q = k.pop("q")
        ID = k.pop("ID")

        try:
            r = f(*a,**k)
            q.put({"result": "success",
                   "ID": ID,
                   "value": r})
        except Exception as e:
            q.put({"result": "failure",
                   "exception": e,
                   "stacktrace": traceback.format_exc(),
                   "ID": ID})
            return
    return _f

def solveForTask_ocaml(_=None,
                       elapsedTime=0.,
                       g=None, task=None,
                       lowerBound=None, upperBound=None, budgetIncrement=None,
                       timeout=None,
                       likelihoodModel=None, # FIXME: unused
                       evaluationTimeout=None, maximumFrontier=None):
    import json
    message = {"DSL": {"logVariable": g.logVariable,
                       "productions": [ {"expression": str(p), "logProbability": l}
                                            for l,_,p in g.productions ]},
               "examples": [{"inputs": list(xs), "output": y} for xs,y in task.examples ],
               "programTimeout": evaluationTimeout,
               # "solverTimeout": max(int(timeout + 0.5), 1),
               "maximumFrontier": maximumFrontier,
               "name": task.name,
               "lowerBound": lowerBound,
               "upperBound": upperBound,
               "budgetIncrement": budgetIncrement,
               "verbose": False}
    if hasattr(task, 'BIC'):
        message["parameterPenalty"] = task.BIC*math.log(len(task.examples))
    if hasattr(task, 'likelihoodThreshold') and task.likelihoodThreshold is not None:
        message["lossThreshold"] = -task.likelihoodThreshold
    if hasattr(task, 'maxParameters') and task.maxParameters is not None:
        message["maxParameters"] = task.maxParameters

    message = json.dumps(message)
    # with open("pipe", "w") as f:
        # f.write(message)
    try:
        response, error = command.run(msg=message,
                                      timeout=max(int(timeout + 0.5), 1))
        response = json.loads(response)
    except OSError as exc:
        raise exc

    pc = response["programCount"]
    # Remove all entries that do not type correctly
    # This can occur because the solver tries to infer the type
    # Sometimes it infers a type that is too general
    response = [r for r in response["solutions"] if Program.parse(r["program"]).canHaveType(task.request) ]

    frontier = Frontier([FrontierEntry(program=p,
                                       logLikelihood=e["logLikelihood"],
                                       logPrior=g.logLikelihood(task.request, p))
                         for e in response
                         for p in [Program.parse(e["program"])] ],
                        task=task)

    if frontier.empty: searchTime = None
    else: searchTime = min(e["time"] for e in response) + elapsedTime

    return frontier, searchTime, pc

def solveForTask_pypy(_=None,
                      elapsedTime=0.,
                      g=None, task=None,
                      lowerBound=None, upperBound=None, budgetIncrement=None,
                      timeout=None,
                      likelihoodModel=None,
                      evaluationTimeout=None, maximumFrontier=None):
    return callCompiled(enumerateForTask,
                        g,task,likelihoodModel,
                        timeout=timeout,
                        evaluationTimeout=evaluationTimeout,
                        maximumFrontier=maximumFrontier,
                        budgetIncrement=budgetIncrement,
                        lowerBound=lowerBound,
                        upperBound=upperBound)

def solveForTask_python(_=None,
                        elapsedTime=0.,
                        g=None, task=None,
                        lowerBound=None, upperBound=None, budgetIncrement=None,
                        timeout=None,
                        likelihoodModel=None,
                        evaluationTimeout=None, maximumFrontier=None):
    return enumerateForTask(g,task,likelihoodModel,
                            timeout=timeout,
                            evaluationTimeout=evaluationTimeout,
                            maximumFrontier=maximumFrontier,
                            budgetIncrement=budgetIncrement,
                            lowerBound=lowerBound, upperBound=upperBound)

#from luke    
def enumerateNetwork(network, tasks_features, likelihoodModel, solver=None,
                       frontierSize=None,
                       enumerationTimeout=None,
                       CPUs=1,
                       maximumFrontier=None,
                       verbose=True,
                       evaluationTimeout=None):
    from time import time
    
    start = time()

    chunk_size = int(math.ceil(len(tasks_features) / CPUs)) if int(math.ceil(len(tasks_features) / CPUs)) > 0 else 1
    eprint("enumerateNetwork with", chunk_size, "tasks per cpu")

    chunked_tasks_features = [tasks_features[i:i + chunk_size] for i in range(0, len(tasks_features), chunk_size)]
    

    #TODO, enumerateNetworkForTasks
    frontierss = parallelMap(CPUs,            
                            lambda cpu_idx__tasks_features: enumerateNetworkForTasks(cpu_idx__tasks_features[0], network, cpu_idx__tasks_features[1],
                                                                     likelihoodModel=likelihoodModel, #this may break
                                                                     frontierSize=frontierSize,
                                                                     timeout=enumerationTimeout,
                                                                     evaluationTimeout = evaluationTimeout,
                                                                     verbose=verbose,
                                                                     maximumFrontier=maximumFrontier),
                            list(zip(list(range(len(chunked_tasks_features))), chunked_tasks_features)),
                            chunksize=1)
    frontiers = [frontier for frontiers in frontierss for frontier in frontiers] #wtf is happening
    # if verbose:
    #     eprint("Enumerated %d frontiers in time %f"%(len(), time() - start))
    return frontiers

class EnumerationTimeout(Exception): pass

#from luke
def enumerateNetworkForTasks(cpu_idx, network, tasks_features, likelihoodModel=None,
                     verbose=False,
                     timeout=None,
                     evaluationTimeout=None,
                     frontierSize=None,
                     maximumFrontier = 10**2):
    assert likelihoodModel is not None
    assert network is not None

    assert (timeout is not None) or (frontierSize is not None), \
        "enumerateForTask: You must provide either a timeout or a frontier size."
    eprint("(%d)"%cpu_idx, "enumerateNetworkForTasks")

    from time import time
    def timeoutCallBack(_1,_2): raise EnumerationTimeout()
    if timeout is not None:
        if verbose: eprint("Alarming timeout for",timeout,"for task [task undefined for now]")
        signal.signal(signal.SIGALRM, timeoutCallBack)
        signal.alarm(timeout)
    
    frontiers = []
    for task, features in tasks_features:
        frontier = []
        starting = time()
        # previousBudget = 0.
        # budget = previousBudget + budgetIncrement

        try:
            totalNumberOfPrograms = 0

            

            seen_proposals = set()
            new_proposals_scores = set()
            numberOfPrograms = 0
            numberOfHits = 0

            for i in range(50):
                random.shuffle(features)
                #inputs = [input for (input, output) in features[:4]]
                outputs = [output for output in features[:5]] #changed from 4 to 5
                #this line 
                samples, scores = network.sampleAndScore([outputs]*100)
                new_proposals_scores = [(tuple(samples[i]), scores[i]) for i in range(len(samples)) if tuple(samples[i]) not in seen_proposals]
                seen_proposals = seen_proposals | set(x[0] for x in new_proposals_scores)

                for sample, prior in new_proposals_scores:
                    try:
                        #eprint("untokenized program:", sample)
                        p = untokeniseProgram(sample)
                        if not isinstance(p, pregex.Pregex): continue

                        #likelihood = task.logLikelihood(p, timeout=evaluationTimeout) #TODO: change this
                        #eprint("tokenized program:", p)
                        _, likelihood = likelihoodModel.score(p, task)
                        eprint("sampled an actual program")
                    except ParseFailure: continue
                    except RunFailure: continue #Happens during likelihood evaluation for e.g. (lambda $3)
                    
                    numberOfPrograms += 1

                    if valid(likelihood):
                        if verbose:
                            eprint("(%d)"%cpu_idx, "Hit",task.name,"with the program",p,"which has prior",prior,"after",time() - starting,"seconds using RobustFill model")
                        frontier.append(FrontierEntry(program = p,
                                                      logPrior = prior,
                                                      logLikelihood = likelihood))
                        numberOfHits += 1

                    # If the alarm is triggered during evaluation,
                    # it will be caught by the catchall exception handler
                    # And so we have to time ourselves out
                    if timeout is not None and time() - starting > timeout:
                        signal.alarm(0)
                        raise EnumerationTimeout
            if verbose:
                eprint("(%d)"%cpu_idx, "enumerated: %d samples, %d programs, %d hits" % (len(seen_proposals), numberOfPrograms, numberOfHits))
                
                # previousBudget = budget
                # budget += budgetIncrement
                # totalNumberOfPrograms += numberOfPrograms
                # if verbose:
                #     eprint("\tTotal elapsed time: %d seconds. Total number of programs evaluated: %d. Task: %s."% \
                #            (time() - starting, totalNumberOfPrograms, task))
                # if frontierSize is not None and totalNumberOfPrograms > frontierSize: break
        except EnumerationTimeout:
            if verbose:
                eprint("Timeout triggered after",time() - starting,"seconds for task",task)
        signal.alarm(0)

        frontier = Frontier(frontier,
                            task = task).topK(maximumFrontier)
        eprint(frontier.summarize())
        
        frontiers.append(frontier)

    return frontiers



def enumerateForTask(g, task, likelihoodModel, _=None,
                     verbose=True,
                     timeout=None,
                     evaluationTimeout=None,
                     frontierSize=None,
                     lowerBound=0.,
                     upperBound=100.,
                     budgetIncrement=1.0, maximumFrontier=10**2):
    assert (timeout is not None) or (frontierSize is not None), \
        "enumerateForTask: You must provide either a timeout or a frontier size."

    from time import time

    timeUntilFirstSolution = None
    frontier = []
    starting = time()
    previousBudget = lowerBound
    budget = lowerBound + budgetIncrement
    try:
        totalNumberOfPrograms = 0
        while len(frontier) < maximumFrontier:
            numberOfPrograms = 0
            for prior,_,p in g.enumeration(Context.EMPTY, [], task.request,
                                           maximumDepth=99,
                                           upperBound=budget,
                                           lowerBound=previousBudget):
                descriptionLength = -prior
                # Shouldn't see it on this iteration
                assert descriptionLength <= budget
                # Should already have seen it
                assert descriptionLength > previousBudget

                numberOfPrograms += 1
                totalNumberOfPrograms += 1

                success, likelihood = likelihoodModel.score(p, task)
                if success:
                    if verbose:
                        eprint("Hit",task.name,"with the program",p,"which has prior",prior,"after",time() - starting,"seconds")
                    if frontier == []: timeUntilFirstSolution = time() - starting
                    frontier.append(FrontierEntry(program=p,
                                                  logPrior=prior,
                                                  logLikelihood=likelihood))

                if timeout is not None and time() - starting > timeout:
                    raise EnumerationTimeout
            if verbose:
                eprint("Enumerated %d programs of satisfying:"%(numberOfPrograms),
                       "%d < MDL <= %d."%(int(previousBudget),int(budget)))

            previousBudget = budget
            budget += budgetIncrement
            if verbose:
                eprint("\tTotal elapsed time: %d seconds. Total number of programs evaluated: %d. Task: %s."% \
                       (time() - starting, totalNumberOfPrograms, task))
            if frontierSize is not None and totalNumberOfPrograms > frontierSize: break
            if budget > upperBound: break
    except EnumerationTimeout:
        if verbose:
            eprint("Timeout triggered after",time() - starting,"seconds for task",task)

    frontier = Frontier(frontier,
                        task=task).topK(maximumFrontier)

    return frontier, timeUntilFirstSolution, numberOfPrograms

def solveSingleTask(grammar, task, maximumBudget=15):
    if isinstance(task, DifferentiableTask):
        rememberOld = True
        history = set([])
    else: rememberOld = False
    for budget in range(2, maximumBudget):
        for _,_,p in grammar.enumeration(Context.EMPTY, [], task.request, budget):
            if rememberOld:
                if p in history: continue
                history.add(p)
            l = task.logLikelihood(p)
            if valid(l): return l,p
    return None

def benchmarkSynthesisTimes(result, tasks, _=None, timeout=None, CPUs=None):
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

    from likelihoodModel import AllOrNothingLikelihoodModel
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
                            grammar, task, AllOrNothingLikelihoodModel,
                            maximumFrontier=1,
                            timeout=timeout-elapsed)
    dt = time() - startTime
    if dt > timeout or len(frontier) == 0: return None
    l = solution.entries[0].logLikelihood
    p = solution.entries[0].program
    eprint("Solved",task,"w/",p,"(log likelihood of task given program:",l,").","in time",dt)
    return dt,l

