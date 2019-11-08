from dreamcoder.likelihoodModel import AllOrNothingLikelihoodModel
from dreamcoder.grammar import *
from dreamcoder.utilities import get_root_dir

import os
import traceback
import subprocess


def multicoreEnumeration(g, tasks, _=None,
                         enumerationTimeout=None,
                         solver='ocaml',
                         CPUs=1,
                         maximumFrontier=None,
                         verbose=True,
                         evaluationTimeout=None,
                         testing=False):
    '''g: Either a Grammar, or a map from task to grammar.
    Returns (list-of-frontiers, map-from-task-to-search-time)'''

    # We don't use actual threads but instead use the multiprocessing
    # library. This is because we need to be able to kill workers.
    #from multiprocess import Process, Queue

    from multiprocessing import Queue

     # everything that gets sent between processes will be dilled
    import dill

    solvers = {"ocaml": solveForTask_ocaml,   
               "pypy": solveForTask_pypy,   
               "python": solveForTask_python}   
    assert solver in solvers, "You must specify a valid solver. options are ocaml, pypy, or python." 

    likelihoodModel = None
    if solver == 'pypy' or solver == 'python':
      # Use an all or nothing likelihood model.
      likelihoodModel = AllOrNothingLikelihoodModel(timeout=evaluationTimeout) 
      
    solver = solvers[solver]

    if not isinstance(g, dict):
        g = {t: g for t in tasks}
    task2grammar = g

    # If we are not evaluating on held out testing tasks:
    # Bin the tasks by request type and grammar
    # If these are the same then we can enumerate for multiple tasks simultaneously
    # If we are evaluating testing tasks:
    # Make sure that each job corresponds to exactly one task
    jobs = {}
    for i, t in enumerate(tasks):
        if testing:
            k = (task2grammar[t], t.request, i)
        else:
            k = (task2grammar[t], t.request)
        jobs[k] = jobs.get(k, []) + [t]

    disableParallelism = len(jobs) == 1
    parallelCallback = launchParallelProcess if not disableParallelism else lambda f, * \
        a, **k: f(*a, **k)
    if disableParallelism:
        eprint("Disabling parallelism on the Python side because we only have one job.")
        eprint("If you are using ocaml, there could still be parallelism.")

    # Map from task to the shortest time to find a program solving it
    bestSearchTime = {t: None for t in task2grammar}

    lowerBounds = {k: 0. for k in jobs}

    frontiers = {t: Frontier([], task=t) for t in task2grammar}

    # For each job we keep track of how long we have been working on it
    stopwatches = {t: Stopwatch() for t in jobs}

    # Map from task to how many programs we enumerated for that task
    taskToNumberOfPrograms = {t: 0 for t in tasks }

    def numberOfHits(f):
        return sum(e.logLikelihood > -0.01 for e in f)

    def budgetIncrement(lb):
        if True:
            return 1.5
        # Very heuristic - not sure what to do here
        if lb < 24.:
            return 1.
        elif lb < 27.:
            return 0.5
        else:
            return 0.25

    def maximumFrontiers(j):
        tasks = jobs[j]
        return {t: maximumFrontier - numberOfHits(frontiers[t]) for t in tasks}

    def allocateCPUs(n, tasks):
        allocation = {t: 0 for t in tasks}
        while n > 0:
            for t in tasks:
                # During testing we use exactly one CPU per task
                if testing and allocation[t] > 0:
                    return allocation
                allocation[t] += 1
                n -= 1
                if n == 0:
                    break
        return allocation

    def refreshJobs():
        for k in list(jobs.keys()):
            v = [t for t in jobs[k]
                 if numberOfHits(frontiers[t]) < maximumFrontier
                 and stopwatches[k].elapsed <= enumerationTimeout]
            if v:
                jobs[k] = v
            else:
                del jobs[k]

    # Workers put their messages in here
    q = Queue()

    # How many CPUs are we using?
    activeCPUs = 0

    # How many CPUs was each job allocated?
    id2CPUs = {}
    # What job was each ID working on?
    id2job = {}
    nextID = 0

    while True:
        refreshJobs()
        # Don't launch a job that we are already working on
        # We run the stopwatch whenever the job is being worked on
        # freeJobs are things that we are not working on but could be
        freeJobs = [j for j in jobs if not stopwatches[j].running
                    and stopwatches[j].elapsed < enumerationTimeout - 0.5]
        if freeJobs and activeCPUs < CPUs:
            # Allocate a CPU to each of the jobs that we have made the least
            # progress on
            freeJobs.sort(key=lambda j: lowerBounds[j])
            # Launch some more jobs until all of the CPUs are being used
            availableCPUs = CPUs - activeCPUs
            allocation = allocateCPUs(availableCPUs, freeJobs)
            for j in freeJobs:
                if allocation[j] == 0:
                    continue
                g, request = j[:2]
                bi = budgetIncrement(lowerBounds[j])
                thisTimeout = enumerationTimeout - stopwatches[j].elapsed
                eprint("(python) Launching %s (%d tasks) w/ %d CPUs. %f <= MDL < %f. Timeout %f." %
                       (request, len(jobs[j]), allocation[j], lowerBounds[j], lowerBounds[j] + bi, thisTimeout))
                stopwatches[j].start()
                parallelCallback(wrapInThread(solver),
                                 q=q, g=g, ID=nextID,
                                 elapsedTime=stopwatches[j].elapsed,
                                 CPUs=allocation[j],
                                 tasks=jobs[j],
                                 lowerBound=lowerBounds[j],
                                 upperBound=lowerBounds[j] + bi,
                                 budgetIncrement=bi,
                                 timeout=thisTimeout,
                                 evaluationTimeout=evaluationTimeout,
                                 maximumFrontiers=maximumFrontiers(j),
                                 testing=testing,
                                 likelihoodModel=likelihoodModel)
                id2CPUs[nextID] = allocation[j]
                id2job[nextID] = j
                nextID += 1

                activeCPUs += allocation[j]
                lowerBounds[j] += bi

        # If nothing is running, and we just tried to launch jobs,
        # then that means we are finished
        if all(not s.running for s in stopwatches.values()):
            break

        # Wait to get a response
        message = Bunch(dill.loads(q.get()))

        if message.result == "failure":
            eprint("PANIC! Exception in child worker:", message.exception)
            eprint(message.stacktrace)
            assert False
        elif message.result == "success":
            # Mark the CPUs is no longer being used and pause the stopwatch
            activeCPUs -= id2CPUs[message.ID]
            stopwatches[id2job[message.ID]].stop()

            newFrontiers, searchTimes, pc = message.value
            for t, f in newFrontiers.items():
                oldBest = None if len(
                    frontiers[t]) == 0 else frontiers[t].bestPosterior
                frontiers[t] = frontiers[t].combine(f)
                newBest = None if len(
                    frontiers[t]) == 0 else frontiers[t].bestPosterior

                taskToNumberOfPrograms[t] += pc

                dt = searchTimes[t]
                if dt is not None:
                    if bestSearchTime[t] is None:
                        bestSearchTime[t] = dt
                    else:
                        # newBest & oldBest should both be defined
                        assert oldBest is not None
                        assert newBest is not None
                        newScore = newBest.logPrior + newBest.logLikelihood
                        oldScore = oldBest.logPrior + oldBest.logLikelihood

                        if newScore > oldScore:
                            bestSearchTime[t] = dt
                        elif newScore == oldScore:
                            bestSearchTime[t] = min(bestSearchTime[t], dt)
        else:
            eprint("Unknown message result:", message.result)
            assert False

    eprint("We enumerated this many programs, for each task:\n\t",
           list(taskToNumberOfPrograms.values()))

    return [frontiers[t] for t in tasks], bestSearchTime

def wrapInThread(f):
    """
    Returns a function that is designed to be run in a thread/threadlike process.
    Result will be either put into the q
    """
    import dill

    def _f(*a, **k):
        q = k.pop("q")
        ID = k.pop("ID")

        try:
            r = f(*a, **k)
            q.put(dill.dumps({"result": "success",
                   "ID": ID,
                   "value": r}))
        except Exception as e:
            q.put(dill.dumps({"result": "failure",
                   "exception": e,
                   "stacktrace": traceback.format_exc(),
                   "ID": ID}))
            return
    return _f


def solveForTask_ocaml(_=None,
                       elapsedTime=0.,
                       CPUs=1,
                       g=None, tasks=None,
                       lowerBound=None, upperBound=None, budgetIncrement=None,
                       timeout=None,
                       testing=None, # FIXME: unused
                       likelihoodModel=None,
                       evaluationTimeout=None, maximumFrontiers=None):

    import json

    def taskMessage(t):
        m = {
            "examples": [{"inputs": list(xs), "output": y} for xs, y in t.examples],
            "name": t.name,
            "request": t.request.json(),
            "maximumFrontier": maximumFrontiers[t]}
        if hasattr(t, "specialTask"):
            special, extra = t.specialTask
            m["specialTask"] = special
            m["extras"] = extra
        return m


    message = {"DSL": g.json(),
               "tasks": [taskMessage(t)
                         for t in tasks],

               "programTimeout": evaluationTimeout,
               "nc": CPUs,
               "timeout": timeout,
               "lowerBound": lowerBound,
               "upperBound": upperBound,
               "budgetIncrement": budgetIncrement,
               "verbose": False,
               "shatter": 5 if len(tasks) == 1 and "turtle" in str(tasks[0].request) else 10}

    if hasattr(tasks[0], 'maxParameters') and tasks[0].maxParameters is not None:
        message["maxParameters"] = tasks[0].maxParameters

    message = json.dumps(message)
    # uncomment this if you want to save the messages being sent to the solver
    

    try:
        solver_file = os.path.join(get_root_dir(), 'solver')
        process = subprocess.Popen(solver_file,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE)
        response, error = process.communicate(bytes(message, encoding="utf-8"))
        response = json.loads(response.decode("utf-8"))
    except OSError as exc:
        raise exc

    except:
        print("response:", response)
        print("error:", error)
        with open("message", "w") as f:
            f.write(message)
        print("message,", message)
        assert False, "MAX RAISE"


    pc = response.get("number_enumerated",0)  # TODO
    frontiers = {}
    searchTimes = {}
    for t in tasks:
        solutions = response[t.name]
        frontier = Frontier([FrontierEntry(program=p,
                                           logLikelihood=e["logLikelihood"],
                                           logPrior=g.logLikelihood(t.request, p))
                             for e in solutions
                             for p in [Program.parse(e["program"])]],
                            task=t)
        frontiers[t] = frontier
        if frontier.empty:
            searchTimes[t] = None
        # This is subtle:
        # The search time we report is actually not be minimum time to find any solution
        # Rather it is the time to find the MAP solution
        # This is important for regression problems,
        # where we might find something with a good prior but bad likelihood early on,
        # and only later discovered the good high likelihood program
        else:
            searchTimes[t] = min(
                (e["logLikelihood"] + e["logPrior"],
                 e["time"]) for e in solutions)[1] + elapsedTime

    return frontiers, searchTimes, pc

def solveForTask_pypy(_=None,
                      elapsedTime=0.,
                      g=None, task=None,
                      lowerBound=None, upperBound=None, budgetIncrement=None,
                      timeout=None,
                      likelihoodModel=None,
                      evaluationTimeout=None, maximumFrontier=None, testing=False):
    return callCompiled(enumerateForTasks,
                        g, tasks, likelihoodModel,
                        timeout=timeout,
                        testing=testing,
                        elapsedTime=elapsedTime,
                        evaluationTimeout=evaluationTimeout,
                        maximumFrontiers=maximumFrontiers,
                        budgetIncrement=budgetIncrement,
                        lowerBound=lowerBound, upperBound=upperBound)

def solveForTask_python(_=None,
                        elapsedTime=0.,
                        g=None, tasks=None,
                        lowerBound=None, upperBound=None, budgetIncrement=None,
                        timeout=None,
                        CPUs=1,
                        likelihoodModel=None,
                        evaluationTimeout=None, maximumFrontiers=None, testing=False):
    return enumerateForTasks(g, tasks, likelihoodModel,
                             timeout=timeout,
                             testing=testing,
                             elapsedTime=elapsedTime,
                             evaluationTimeout=evaluationTimeout,
                             maximumFrontiers=maximumFrontiers,
                             budgetIncrement=budgetIncrement,
                             lowerBound=lowerBound, upperBound=upperBound)


class EnumerationTimeout(Exception):
    pass

def enumerateForTasks(g, tasks, likelihoodModel, _=None,
                      verbose=False,
                      timeout=None,
                      elapsedTime=0.,
                      CPUs=1,
                      testing=False, #unused
                      evaluationTimeout=None,
                      lowerBound=0.,
                      upperBound=100.,
                      budgetIncrement=1.0, maximumFrontiers=None):
    assert timeout is not None, \
        "enumerateForTasks: You must provide a timeout."

    from time import time

    request = tasks[0].request
    assert all(t.request == request for t in tasks), \
        "enumerateForTasks: Expected tasks to all have the same type"

    maximumFrontiers = [maximumFrontiers[t] for t in tasks]
    # store all of the hits in a priority queue
    # we will never maintain maximumFrontier best solutions
    hits = [PQ() for _ in tasks]

    starting = time()
    previousBudget = lowerBound
    budget = lowerBound + budgetIncrement
    try:
        totalNumberOfPrograms = 0
        while time() < starting + timeout and \
                any(len(h) < mf for h, mf in zip(hits, maximumFrontiers)) and \
                budget <= upperBound:
            numberOfPrograms = 0

            for prior, _, p in g.enumeration(Context.EMPTY, [], request,
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

                for n in range(len(tasks)):
                    task = tasks[n]

                    #Warning:changed to max's new likelihood model situation
                    #likelihood = task.logLikelihood(p, evaluationTimeout)
                    #if invalid(likelihood):
                        #continue
                    success, likelihood = likelihoodModel.score(p, task)
                    if not success:
                        continue
                        
                    dt = time() - starting + elapsedTime
                    priority = -(likelihood + prior)
                    hits[n].push(priority,
                                 (dt, FrontierEntry(program=p,
                                                    logLikelihood=likelihood,
                                                    logPrior=prior)))
                    if len(hits[n]) > maximumFrontiers[n]:
                        hits[n].popMaximum()

                if timeout is not None and time() - starting > timeout:
                    raise EnumerationTimeout

            previousBudget = budget
            budget += budgetIncrement

            if budget > upperBound:
                break
    except EnumerationTimeout:
        pass
    frontiers = {tasks[n]: Frontier([e for _, e in hits[n]],
                                    task=tasks[n])
                 for n in range(len(tasks))}
    searchTimes = {
        tasks[n]: None if len(hits[n]) == 0 else \
        min(t for t,_ in hits[n]) for n in range(len(tasks))}

    return frontiers, searchTimes, totalNumberOfPrograms





