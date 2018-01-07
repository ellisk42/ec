from frontier import *
from task import *
from type import *
from program import *
from grammar import *

def enumerateFrontiers(g, frontierSize, tasks):
    from time import time
    requests = { t.request for t in tasks }
    frontiers = {}
    start = time()
    for request in requests:
        frontiers[request] = iterativeDeepeningEnumeration(g, request, frontierSize)
        
    totalNumberOfPrograms = sum(len(f) for f in frontiers.values())
    print "Enumerated %d frontiers with %d total programs in time %fsec"%(len(frontiers),totalNumberOfPrograms,time() - start)
    start = time()
    frontiers = [
        Frontier([ FrontierEntry(program,
                                 logPrior = logPrior,
                                 logLikelihood = task.logLikelihood(program))
                   for logPrior,program in frontiers[task.request] ],
                 task = task).\
        removeZeroLikelihood()
        for task in tasks ]
    
    dt = time() - start
    print "Scored frontiers in time %fsec (%f/program)"%(dt,dt/totalNumberOfPrograms)

    return frontiers

def iterativeDeepeningEnumeration(g, request, frontierSize,
                                  budget = 2.0, budgetIncrement = 1.0):
    frontier = []
    while len(frontier) < frontierSize:
        frontier = [ (l,p) for l,_,p in enumeration(g, Context.EMPTY, [], request, budget) ]
        budget += budgetIncrement
    print "Enumerated up to %f nats"%(budget - budgetIncrement)
    return frontier

def enumeration(g, context, environment, request, budget):
    if budget <= 0: return
    if request.name == ARROW:
        v = request.arguments[0]
        for l,newContext,b in enumeration(g, context, [v] + environment, request.arguments[1], budget):
            yield (l, newContext, Abstraction(b))

    else:
        candidates = []
        for l,t,p in g.productions:
            try:
                newContext, t = t.instantiate(context)
                newContext = newContext.unify(t.returns(), request)
                candidates.append((l,newContext,
                                   t.apply(newContext),
                                   p))
            except UnificationFailure: continue
        for j,t in enumerate(environment):
            try:
                newContext = context.unify(t.returns(), request)
                candidates.append((g.logVariable,newContext,
                                   t.apply(newContext),
                                   Index(j)))
            except UnificationFailure: continue
        
        z = math.log(sum(math.exp(candidate[0]) for candidate in candidates))
        
        for (l,newContext,t,p) in candidates:
            l -= z
            xs = t.functionArguments()
            for result in enumerateApplication(g, newContext, environment,
                                               p, l, xs, budget + l):
                yield result

def enumerateApplication(g, context, environment,
                         function, functionLikelihood,
                         argumentRequests, budget):
    if argumentRequests == []: yield (functionLikelihood, context, function)
    else:
        firstRequest = argumentRequests[0].apply(context)
        laterRequests = argumentRequests[1:]
        for firstLikelihood, newContext, firstArgument in enumeration(g, context, environment, firstRequest, budget):
            newFunction = Application(function, firstArgument)
            for result in enumerateApplication(g, newContext, environment,
                                               newFunction, functionLikelihood + firstLikelihood,
                                               laterRequests,
                                               budget + firstLikelihood):
                yield result
            
    
