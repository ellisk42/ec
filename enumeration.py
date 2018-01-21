from utilities import eprint
from frontier import *
from task import *
from type import *
from program import *
from grammar import *


def enumerateFrontiers(g, frontierSize, tasks, CPUs=1, maximumFrontier=None):
    '''g: Either a Grammar, or a map from task to grammar.'''
    from time import time

    frontiers = {}

    start = time()
    if isinstance(g, Grammar):
        for request in { t.request for t in tasks }:
            frontier = iterativeDeepeningEnumeration(g, request, frontierSize)
            frontiers[request] = frontier
        totalNumberOfPrograms = sum(len(f) for f in frontiers.values())
        totalNumberOfFrontiers = len(frontiers)

        frontiers = {t: frontiers[t.request] for t in tasks}
    else:
        frontiers = dict(parallelMap(
            CPUs,
            lambda t: (t, iterativeDeepeningEnumeration(g[t], t.request, frontierSize)),
            tasks))
        totalNumberOfPrograms = sum(len(f) for f in frontiers.values())
        totalNumberOfFrontiers = len(frontiers)

    eprint("Enumerated %d frontiers with %d total programs in time %fsec" %
           (totalNumberOfFrontiers, totalNumberOfPrograms, time() - start))

    start = time()
    # We split up the likelihood calculation and the frontier construction
    # This is so we do not have to serialize and deserialize a bunch of programs
    # programLikelihoods: [ {indexInfrontiers[task]: likelihood} (for each task)]
    programLikelihoods = parallelMap(CPUs, lambda task:
                                     {j: logLikelihood
                                      for j, (_, program) in enumerate(frontiers[task])
                                      for logLikelihood in [task.logLikelihood(program)]
                                      if valid(logLikelihood)},
                                     tasks)

    frontiers = constructFrontiers(frontiers, programLikelihoods, tasks, maximumFrontier)

    dt = time() - start
    eprint("Scored frontiers in time %fsec (%f/program)" % (dt, dt / totalNumberOfPrograms))

    return frontiers


def iterativeDeepeningEnumeration(g, request, frontierSize, budget=2.0, budgetIncrement=1.0):
    frontier = []
    while len(frontier) < frontierSize:
        frontier = [(l, p) for l, _, p in enumeration(g, Context.EMPTY, [], request, budget)]
        budget += budgetIncrement
    #eprint("Enumerated up to %f nats"%(budget - budgetIncrement))
    # This will trim the frontier to be exactly frontierSize
    # Not sure whether we want to do this
    #return sorted(frontier, key=lambda (l, p): l, reverse=True)[:frontierSize]
    return frontier


def enumeration(g, context, environment, request, budget):
    if budget <= 0:
        return
    if request.name == ARROW:
        v = request.arguments[0]
        for l, newContext, b in enumeration(g, context, [v] + environment,
                                            request.arguments[1], budget):
            yield l, newContext, Abstraction(b)

    else:
        candidates = []
        for l, t, p in g.productions:
            try:
                newContext, t = t.instantiate(context)
                newContext = newContext.unify(t.returns(), request)
                candidates.append((l, newContext, t.apply(newContext), p))
            except UnificationFailure:
                continue
        for j, t in enumerate(environment):
            try:
                newContext = context.unify(t.returns(), request)
                candidates.append((g.logVariable, newContext,
                                   t.apply(newContext), Index(j)))
            except UnificationFailure:
                continue

        z = lse([candidate[0] for candidate in candidates])

        for l, newContext, t, p in candidates:
            l -= z
            xs = t.functionArguments()
            for result in enumerateApplication(g, newContext, environment, p, l, xs, budget + l):
                yield result


def enumerateApplication(g, context, environment,
                         function, functionLikelihood, argumentRequests, budget):
    if argumentRequests == []:
        yield functionLikelihood, context, function
    else:
        argRequest = argumentRequests[0].apply(context)
        laterRequests = argumentRequests[1:]
        for argL, newContext, arg in enumeration(g, context, environment, argRequest, budget):
            newFunction = Application(function, arg)
            for result in enumerateApplication(g, newContext, environment, newFunction,
                                               functionLikelihood + argL,
                                               laterRequests, budget + argL):
                yield result


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
