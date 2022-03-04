from frozendict import frozendict
from collections import defaultdict

from dreamcoder.frontier import *
from dreamcoder.program import *
from dreamcoder.type import *
from dreamcoder.utilities import *

import time

import itertools

class GrammarFailure(Exception):
    pass

class SketchEnumerationFailure(Exception):
    pass

class NoCandidates(Exception):
    pass


class Grammar(object):
    def __init__(self, logVariable, productions, continuationType=None):
        self.logVariable = logVariable
        self.productions = productions

        self.continuationType = continuationType

        self.expression2likelihood = dict((p, l) for l, _, p in productions)
        self.expression2likelihood[Index(0)] = self.logVariable

    def randomWeights(self, r):
        """returns a new grammar with random weights drawn from r. calls `r` w/ old weight"""
        return Grammar(logVariable=r(self.logVariable),
                       productions=[(r(l),t,p)
                                    for l,t,p in self.productions ],
                       continuationType=self.continuationType)

    def strip_primitive_values(self):
        return Grammar(logVariable=self.logVariable,
                       productions=[(l,t,strip_primitive_values(p))
                                    for l,t,p in self.productions ],
                       continuationType=self.continuationType)

    def unstrip_primitive_values(self):
        return Grammar(logVariable=self.logVariable,
                       productions=[(l,t,unstrip_primitive_values(p))
                                    for l,t,p in self.productions ],
                       continuationType=self.continuationType)

    def __setstate__(self, state):
        """
        Legacy support for loading grammar objects without the imperative type filled in
        """
        assert 'logVariable' in state
        assert 'productions' in state
        if 'continuationType' in state:
            continuationType = state['continuationType']
        else:
            if any( 'turtle' in str(t) for l,t,p in state['productions'] ):
                continuationType = baseType("turtle")
            elif any( 'tower' in str(t) for l,t,p in state['productions'] ):
                continuationType = baseType("tower")
            else:
                continuationType = None
                
        self.__init__(state['logVariable'], state['productions'], continuationType=continuationType)

    @staticmethod
    def fromProductions(productions, logVariable=0.0, continuationType=None):
        """Make a grammar from primitives and their relative logpriors."""
        return Grammar(logVariable, [(l, p.infer(), p)
                                     for l, p in productions],
                       continuationType=continuationType)

    @staticmethod
    def uniform(primitives, continuationType=None):
        return Grammar(0.0, [(0.0, p.infer(), p) for p in primitives], continuationType=continuationType)

    def __len__(self): return len(self.productions)

    def __str__(self):
        def productionKey(xxx_todo_changeme):
            (l, t, p) = xxx_todo_changeme
            return not isinstance(p, Primitive), l is not None and -l
        if self.continuationType is not None:
            lines = ["continuation : %s"%self.continuationType]
        else:
            lines = []
        lines += ["%f\tt0\t$_" % self.logVariable]
        for l, t, p in sorted(self.productions, key=productionKey):
            if l is not None:
                l = "%f\t%s\t%s" % (l, t, p)
            else:
                l = "-Inf\t%s\t%s" % (t, p)
            if not t.isArrow() and isinstance(p, Invented):
                try:
                    l += "\teval = %s" % (p.evaluate([]))
                except BaseException:
                    pass

            lines.append(l)
        return "\n".join(lines)

    def json(self):
        j = {"logVariable": self.logVariable,
             "productions": [{"expression": str(p), "logProbability": l}
                             for l, _, p in self.productions]}
        if self.continuationType is not None:
            j["continuationType"] = self.continuationType.json()
        return j

    def _immutable_code(self): return self.logVariable, tuple(self.productions)

    def __eq__(self, o): return self._immutable_code() == o._immutable_code()

    def __ne__(self, o): return not (self == o)

    def __hash__(self): return hash(self._immutable_code())

    @property
    def primitives(self):
        return [p for _, _, p in self.productions]

    def removeProductions(self, ps):
        return Grammar(
            self.logVariable, [
                (l, t, p) for (
                    l, t, p) in self.productions if p not in ps],
            continuationType=self.continuationType)

    def buildCandidates(self, request, context, environment,
                        # Should the log probabilities be normalized?
                        normalize=True,
                        # Should be returned a table mapping primitives to
                        # their candidate entry?
                        returnTable=False,
                        # Should we return probabilities vs log probabilities?
                        returnProbabilities=False,
                        # Must be a leaf (have no arguments)?
                        mustBeLeaf=False):
        """Primitives that are candidates for being used given a requested type
        If returnTable is false (default): returns [((log)likelihood, tp, primitive, context)]
        if returntable is true: returns {primitive: ((log)likelihood, tp, context)}"""
        if returnProbabilities:
            assert normalize

        candidates = []
        variableCandidates = []
        for l, t, p in self.productions:
            try:
                newContext, t = t.instantiate(context)
                newContext = newContext.unify(t.returns(), request)
                t = t.apply(newContext)
                if mustBeLeaf and t.isArrow():
                    continue
                candidates.append((l, t, p, newContext))
            except UnificationFailure:
                continue
        for j, t in enumerate(environment):
            try:
                newContext = context.unify(t.returns(), request)
                t = t.apply(newContext)
                if mustBeLeaf and t.isArrow():
                    continue
                variableCandidates.append((t, Index(j), newContext))
            except UnificationFailure:
                continue

        if self.continuationType == request:
            terminalIndices = [v.i for t,v,k in variableCandidates if not t.isArrow()]
            if terminalIndices:
                smallestIndex = Index(min(terminalIndices))
                variableCandidates = [(t,v,k) for t,v,k in variableCandidates
                                      if t.isArrow() or v == smallestIndex]
            
        candidates += [(self.logVariable - log(len(variableCandidates)), t, p, k)
                       for t, p, k in variableCandidates]
        if candidates == []:
            raise NoCandidates()
        #eprint("candidates inside buildCandidates before norm:")
        #eprint(candidates)

        if normalize:
            z = lse([l for l, t, p, k in candidates])
            if returnProbabilities:
                candidates = [(exp(l - z), t, p, k)
                              for l, t, p, k in candidates]
            else:
                candidates = [(l - z, t, p, k) for l, t, p, k in candidates]

        #eprint("candidates inside buildCandidates after norm:")
        #eprint(candidates)

        if returnTable:
            return {p: (l, t, k) for l, t, p, k in candidates}
        else:
            return candidates


    def sample(self, request, maximumDepth=6, maxAttempts=None):
        attempts = 0

        while True:
            try:
                _, e = self._sample(
                    request, Context.EMPTY, [], maximumDepth=maximumDepth)
                return e
            except NoCandidates:
                if maxAttempts is not None:
                    attempts += 1
                    if attempts > maxAttempts:
                        return None
                continue

    def _sample(self, request, context, environment, maximumDepth):
        if request.isArrow():
            context, expression = self._sample(
                request.arguments[1], context, [
                    request.arguments[0]] + environment, maximumDepth)
            return context, Abstraction(expression)

        candidates = self.buildCandidates(request, context, environment,
                                          normalize=True,
                                          returnProbabilities=True,
                                          # Force it to terminate in a
                                          # leaf; a primitive with no
                                          # function arguments
                                          mustBeLeaf=maximumDepth <= 1)
        #eprint("candidates:")
        #eprint(candidates)
        newType, chosenPrimitive, context = sampleDistribution(candidates)

        # Sample the arguments
        xs = newType.functionArguments()
        returnValue = chosenPrimitive

        for x in xs:
            x = x.apply(context)
            context, x = self._sample(x, context, environment, maximumDepth - 1)
            returnValue = Application(returnValue, x)

        return context, returnValue

    def likelihoodSummary(self, context, environment, request, expression, silent=False):
        if request.isArrow():
            if not isinstance(expression, Abstraction):
                if not silent:
                    eprint("Request is an arrow but I got", expression)
                return context, None
            return self.likelihoodSummary(context,
                                          [request.arguments[0]] + environment,
                                          request.arguments[1],
                                          expression.body,
                                          silent=silent)
        # Build the candidates
        candidates = self.buildCandidates(request, context, environment,
                                          normalize=False,
                                          returnTable=True)

        # A list of everything that would have been possible to use here
        possibles = [p for p in candidates.keys() if not p.isIndex]
        numberOfVariables = sum(p.isIndex for p in candidates.keys())
        if numberOfVariables > 0:
            possibles += [Index(0)]

        f, xs = expression.applicationParse()

        if f not in candidates:
            if self.continuationType is not None and f.isIndex:
                ls = LikelihoodSummary()
                ls.constant = NEGATIVEINFINITY
                return ls
            
            if not silent:
                eprint(f, "Not in candidates")
                eprint("Candidates is", candidates)
                #eprint("grammar:", grammar.productions)
                eprint("request is", request)
                eprint("xs", xs)
                eprint("environment", environment)
                assert False
            return context, None

        thisSummary = LikelihoodSummary()
        thisSummary.record(f, possibles,
                           constant= -math.log(numberOfVariables) if f.isIndex else 0)

        _, tp, context = candidates[f]
        argumentTypes = tp.functionArguments()
        if len(xs) != len(argumentTypes):
            eprint("PANIC: not enough arguments for the type")
            eprint("request", request)
            eprint("tp", tp)
            eprint("expression", expression)
            eprint("xs", xs)
            eprint("argumentTypes", argumentTypes)
            # This should absolutely never occur
            raise GrammarFailure((context, environment, request, expression))

        for argumentType, argument in zip(argumentTypes, xs):
            argumentType = argumentType.apply(context)
            context, newSummary = self.likelihoodSummary(
                context, environment, argumentType, argument, silent=silent)
            if newSummary is None:
                return context, None
            thisSummary.join(newSummary)

        return context, thisSummary

    def bestFirstEnumeration(self, request):
        from heapq import heappush, heappop

        pq = []

        def choices(parentCost, xs):
            for c, x in xs:
                heappush(pq, (parentCost + c, x))

        def g(parentCost, request, _=None,
              context=None, environment=[],
              k=None):
            """
            k is a continuation.
            k: Expects to be called with MDL, context, expression.
            """

            assert k is not None
            if context is None:
                context = Context.EMPTY

            if request.isArrow():
                g(parentCost,
                  request.arguments[1],
                  context=context,
                  environment=[request.arguments[0]] + environment,
                    k=lambda MDL,
                    newContext,
                    p: k(MDL,
                         newContext,
                         Abstraction(p)))
            else:
                candidates = self.buildCandidates(request,
                                                  context,
                                                  environment,
                                                  normalize=True,
                                                  returnProbabilities=False,
                                                  returnTable=True)
                choices(parentCost,
                        [(-f_ll_tp_newContext[1][0],
                          lambda: ga(parentCost - f_ll_tp_newContext[1][0],
                                     f_ll_tp_newContext[0],
                                     f_ll_tp_newContext[1][1].functionArguments(),
                                     context=f_ll_tp_newContext[1][2],
                                     environment=environment,
                                     k=k)) for f_ll_tp_newContext in iter(candidates.items())])

        def ga(costSoFar, f, argumentTypes, _=None,
               context=None, environment=None,
               k=None):
            if argumentTypes == []:
                k(costSoFar, context, f)
            else:
                t1 = argumentTypes[0].apply(context)
                g(costSoFar, t1, context=context, environment=environment,
                  k=lambda newCost, newContext, argument:
                  ga(newCost, Application(f, argument), argumentTypes[1:],
                     context=newContext, environment=environment,
                     k=k))

        def receiveResult(MDL, _, expression):
            heappush(pq, (MDL, expression))

        g(0., request, context=Context.EMPTY, environment=[], k=receiveResult)
        frontier = []
        while len(frontier) < 10**3:
            MDL, action = heappop(pq)
            if isinstance(action, Program):
                expression = action
                frontier.append(expression)
                #eprint("Enumerated program",expression,-MDL,self.closedLogLikelihood(request, expression))
            else:
                action()

    def closedLikelihoodSummary(self, request, expression, silent=False):
        try:
            context, summary = self.likelihoodSummary(Context.EMPTY, [], request, expression, silent=silent)
        except GrammarFailure as e:
            failureExport = 'failures/grammarFailure%s.pickle' % (
                time.time() + getPID())
            eprint("PANIC: Grammar failure, exporting to ", failureExport)
            with open(failureExport, 'wb') as handle:
                pickle.dump((e, self, request, expression), handle)
            assert False

        return summary

    def logLikelihood(self, request, expression):
        summary = self.closedLikelihoodSummary(request, expression)
        if summary is None:
            eprint(
                "FATAL: program [ %s ] does not have a likelihood summary." %
                expression, "r = ", request, "\n", self)
            assert False
        return summary.logLikelihood(self)

    def rescoreFrontier(self, frontier):
        return Frontier([FrontierEntry(e.program,
                                       logPrior=self.logLikelihood(frontier.task.request, e.program),
                                       logLikelihood=e.logLikelihood)
                         for e in frontier],
                        frontier.task)

    def productionUses(self, frontiers):
        """Returns the expected number of times that each production was used. {production: expectedUses}"""
        frontiers = [self.rescoreFrontier(f).normalize()
                     for f in frontiers if not f.empty]
        uses = {p: 0. for p in self.primitives}
        for f in frontiers:
            for e in f:
                summary = self.closedLikelihoodSummary(f.task.request,
                                                       e.program)
                for p, u in summary.uses:
                    uses[p] += u * math.exp(e.logPosterior)
        return uses

    def insideOutside(self, frontiers, pseudoCounts, iterations=1):
        # Replace programs with (likelihood summary, uses)
        frontiers = [ Frontier([ FrontierEntry((summary, summary.toUses()),
                                               logPrior=summary.logLikelihood(self),
                                               logLikelihood=e.logLikelihood)
                                 for e in f
                                 for summary in [self.closedLikelihoodSummary(f.task.request, e.program)] ],
                               task=f.task)
                      for f in frontiers ]

        g = self
        for i in range(iterations):
            u = Uses()
            for f in frontiers:
                f = f.normalize()
                for e in f:
                    _, eu = e.program
                    u += math.exp(e.logPosterior) * eu

            lv = math.log(u.actualVariables + pseudoCounts) - \
                 math.log(u.possibleVariables + pseudoCounts)
            g = Grammar(lv,
                        [ (math.log(u.actualUses.get(p,0.) + pseudoCounts) - \
                           math.log(u.possibleUses.get(p,0.) + pseudoCounts),
                           t,p)
                          for _,t,p in g.productions ],
                        continuationType=self.continuationType)
            if i < iterations - 1:
                frontiers = [Frontier([ FrontierEntry((summary, uses),
                                                      logPrior=summary.logLikelihood(g),
                                                      logLikelihood=e.logLikelihood)
                                        for e in f
                                        for (summary, uses) in [e.program] ],
                                      task=f.task)
                             for f in frontiers ]
        return g

    def frontierMDL(self, frontier):
        return max( e.logLikelihood + self.logLikelihood(frontier.task.request, e.program)
                    for e in frontier )                


    def enumeration(self,context,environment,request,upperBound,
                    maximumDepth=20,
                    lowerBound=0.):
        '''Enumerates all programs whose MDL satisfies: lowerBound <= MDL < upperBound'''
        if upperBound < 0 or maximumDepth == 1:
            return

        if request.isArrow():
            v = request.arguments[0]
            for l, newContext, b in self.enumeration(context, [v] + environment,
                                                     request.arguments[1],
                                                     upperBound=upperBound,
                                                     lowerBound=lowerBound,
                                                     maximumDepth=maximumDepth):
                yield l, newContext, Abstraction(b)

        else:
            candidates = self.buildCandidates(request, context, environment,
                                              normalize=True)

            for l, t, p, newContext in candidates:
                mdl = -l
                if not (mdl < upperBound):
                    continue

                xs = t.functionArguments()
                for aL, aK, application in\
                    self.enumerateApplication(newContext, environment, p, xs,
                                              upperBound=upperBound + l,
                                              lowerBound=lowerBound + l,
                                              maximumDepth=maximumDepth - 1):
                    yield aL + l, aK, application

    def enumerateApplication(self, context, environment,
                             function, argumentRequests,
                             # Upper bound on the description length of all of
                             # the arguments
                             upperBound,
                             # Lower bound on the description length of all of
                             # the arguments
                             lowerBound=0.,
                             maximumDepth=20,
                             originalFunction=None,
                             argumentIndex=0):
        if upperBound < 0. or maximumDepth == 1:
            return
        if originalFunction is None:
            originalFunction = function

        if argumentRequests == []:
            if lowerBound <= 0. and 0. < upperBound:
                yield 0., context, function
            else:
                return
        else:
            argRequest = argumentRequests[0].apply(context)
            laterRequests = argumentRequests[1:]
            for argL, newContext, arg in self.enumeration(context, environment, argRequest,
                                                          upperBound=upperBound,
                                                          lowerBound=0.,
                                                          maximumDepth=maximumDepth):
                if violatesSymmetry(originalFunction, arg, argumentIndex):
                    continue

                newFunction = Application(function, arg)
                for resultL, resultK, result in self.enumerateApplication(newContext, environment, newFunction,
                                                                          laterRequests,
                                                                          upperBound=upperBound + argL,
                                                                          lowerBound=lowerBound + argL,
                                                                          maximumDepth=maximumDepth,
                                                                          originalFunction=originalFunction,
                                                                          argumentIndex=argumentIndex + 1):
                    yield resultL + argL, resultK, result

    def sketchEnumeration(self,context,environment,request,sk,upperBound,
                           maximumDepth=20,
                           lowerBound=0.):
        '''Enumerates all sketch instantiations whose MDL satisfies: lowerBound <= MDL < upperBound'''
        if upperBound < 0. or maximumDepth == 1:
            return

        if sk.isHole:
            yield from self.enumeration(context, environment, request, upperBound,
                                        maximumDepth=maximumDepth,
                                        lowerBound=lowerBound)
        elif request.isArrow():
            assert sk.isAbstraction
            v = request.arguments[0]
            for l, newContext, b in self.sketchEnumeration(context, [v] + environment,
                                                           request.arguments[1],
                                                           sk.body,
                                                           upperBound=upperBound,
                                                           lowerBound=lowerBound,
                                                           maximumDepth=maximumDepth):
                yield l, newContext, Abstraction(b)

        else:
            f, xs = sk.applicationParse()
            if f.isIndex:
                ft = environment[f.i].apply(context)
            elif f.isInvented or f.isPrimitive:
                context, ft = f.tp.instantiate(context)
            elif f.isAbstraction:
                assert False, "sketch is not in beta longform"
            elif f.isHole:
                assert False, "hole as function not yet supported"
            elif f.isApplication:
                assert False, "should never happen - bug in applicationParse"
            else: assert False

            try: context = context.unify(ft.returns(), request)                
            except UnificationFailure:
                eprint("Exception: sketch is ill-typed")
                return #so that we can continue evaluating
                # raise SketchEnumerationFailure() #"sketch is ill-typed"
            ft = ft.apply(context)
            argumentRequests = ft.functionArguments()

            assert len(argumentRequests) == len(xs)

            yield from self.sketchApplication(context, environment,
                                              f, xs, argumentRequests,
                                              upperBound=upperBound,
                                              lowerBound=lowerBound,
                                              maximumDepth=maximumDepth - 1)


    def sketchApplication(self, context, environment,
                          function, arguments, argumentRequests,
                          # Upper bound on the description length of all of
                          # the arguments
                          upperBound,
                          # Lower bound on the description length of all of
                          # the arguments
                          lowerBound=0.,
                          maximumDepth=20):
        if upperBound < 0. or maximumDepth == 1:
            return

        if argumentRequests == []:
            if lowerBound <= 0. and 0. < upperBound:
                yield 0., context, function
            else:
                return
        else:
            argRequest = argumentRequests[0].apply(context)
            laterRequests = argumentRequests[1:]
            firstSketch = arguments[0]
            laterSketches = arguments[1:]
            for argL, newContext, arg in self.sketchEnumeration(context, environment, argRequest,
                                                                firstSketch,
                                                                upperBound=upperBound,
                                                                lowerBound=0.,
                                                                maximumDepth=maximumDepth):

                newFunction = Application(function, arg)
                for resultL, resultK, result in self.sketchApplication(newContext, environment, newFunction,
                                                                       laterSketches, laterRequests,
                                                                       upperBound=upperBound + argL,
                                                                       lowerBound=lowerBound + argL,
                                                                       maximumDepth=maximumDepth):

                    yield resultL + argL, resultK, result

    def sketchLogLikelihood(self, request, full, sk, context=Context.EMPTY, environment=[]):
        """
        calculates mdl of full program 'full' from sketch 'sk'
        """
        if sk.isHole:
            _, summary = self.likelihoodSummary(context, environment, request, full)
            if summary is None:
                eprint(
                    "FATAL: program [ %s ] does not have a likelihood summary." %
                    full, "r = ", request, "\n", self)
                assert False
            return summary.logLikelihood(self), context

        elif request.isArrow():
            assert sk.isAbstraction and full.isAbstraction
            #assert sk.f == full.f #is this right? or do i need to recurse?
            v = request.arguments[0]
            return self.sketchLogLikelihood(request.arguments[1], full.body, sk.body, context=context, environment=[v] + environment)

        else:
            sk_f, sk_xs = sk.applicationParse()
            full_f, full_xs = full.applicationParse()
            if sk_f.isIndex:
                assert sk_f == full_f, "sketch and full program don't match on an index"
                ft = environment[sk_f.i].apply(context)
            elif sk_f.isInvented or sk_f.isPrimitive:
                assert sk_f == full_f, "sketch and full program don't match on a primitive"
                context, ft = sk_f.tp.instantiate(context)
            elif sk_f.isAbstraction:
                assert False, "sketch is not in beta longform"
            elif sk_f.isHole:
                assert False, "hole as function not yet supported"
            elif sk_f.isApplication:
                assert False, "should never happen - bug in applicationParse"
            else: assert False

            try: context = context.unify(ft.returns(), request)                
            except UnificationFailure: assert False, "sketch is ill-typed"
            ft = ft.apply(context)
            argumentRequests = ft.functionArguments()

            assert len(argumentRequests) == len(sk_xs) == len(full_xs)  #this might not be true if holes??

            return self.sketchllApplication(context, environment,
                                              sk_f, sk_xs, full_f, full_xs, argumentRequests)

    def sketchllApplication(self, context, environment,
                          sk_function, sk_arguments, full_function, full_arguments, argumentRequests):
        if argumentRequests == []:
                return torch.tensor([0.]).cuda(), context #does this make sense?
        else:
            argRequest = argumentRequests[0].apply(context)
            laterRequests = argumentRequests[1:]

            sk_firstSketch = sk_arguments[0]
            full_firstSketch = full_arguments[0]
            sk_laterSketches = sk_arguments[1:]
            full_laterSketches = full_arguments[1:]

            argL, newContext = self.sketchLogLikelihood(argRequest, full_firstSketch, sk_firstSketch, context=context, environment=environment)

            #finish this...
            sk_newFunction = Application(sk_function, sk_firstSketch)  # is this redundant? maybe 
            full_newFunction = Application(full_function, full_firstSketch)

            resultL, context = self.sketchllApplication(newContext, environment, sk_newFunction, sk_laterSketches,
                                            full_newFunction, full_laterSketches, laterRequests)

            return resultL + argL, context

        
    def enumerateNearby(self, request, expr, distance=3.0):
        """Enumerate programs with local mutations in subtrees with small description length"""
        if distance <= 0:
            yield expr
        else:
            def mutations(tp, loss):
                for l, _, expr in self.enumeration(
                        Context.EMPTY, [], tp, distance - loss):
                    yield expr, l
            yield from Mutator(self, mutations).execute(expr, request)


    def enumerateHoles(self, request, expr, k=3, return_obj=Hole):
        """Enumerate programs with a single hole within mdl distance"""
        #TODO: make it possible to enumerate sketches with multiple holes
        def mutations(tp, loss, is_left_application=False):
            """
            to allow applications lhs to become a hole,  
            remove the condition below and ignore all the is_left_application kwds 
            """
            if not is_left_application: 
                yield return_obj(), 0
        top_k = []
        for expr, l in Mutator(self, mutations).execute(expr, request):
            if len(top_k) > 0:
                i, v = min(enumerate(top_k), key=lambda x:x[1][1])
                if l > v[1]:
                    if len(top_k) >= k:
                        top_k[i] = (expr, l)
                    else:
                        top_k.append((expr, l))
                elif len(top_k) < k:
                    top_k.append((expr, l))
            else:
                top_k.append((expr, l))
        return sorted(top_k, key=lambda x:-x[1])

    def untorch(self):
        return Grammar(self.logVariable.data.tolist()[0], 
                       [ (l.data.tolist()[0], t, p)
                         for l, t, p in self.productions],
                       continuationType=self.continuationType)

class LikelihoodSummary(object):
    '''Summarizes the terms that will be used in a likelihood calculation'''

    def __init__(self):
        self.uses = {}
        self.normalizers = {}
        self.constant = 0.

    def __str__(self):
        return """LikelihoodSummary(constant = %f,
uses = {%s},
normalizers = {%s})""" % (self.constant,
                          ", ".join(
                              "%s: %d" % (k,
                                          v) for k,
                              v in self.uses.items()),
                          ", ".join(
                              "%s: %d" % (k,
                                          v) for k,
                              v in self.normalizers.items()))

    def record(self, actual, possibles, constant=0.):
        # Variables are all normalized to be $0
        if isinstance(actual, Index):
            actual = Index(0)

        # Make it something that we can hash
        possibles = frozenset(sorted(possibles, key=hash))

        self.constant += constant
        self.uses[actual] = self.uses.get(actual, 0) + 1
        self.normalizers[possibles] = self.normalizers.get(possibles, 0) + 1

    def join(self, other):
        self.constant += other.constant
        for k, v in other.uses.items():
            self.uses[k] = self.uses.get(k, 0) + v
        for k, v in other.normalizers.items():
            self.normalizers[k] = self.normalizers.get(k, 0) + v

    def logLikelihood(self, grammar):
        return self.constant + \
            sum(count * grammar.expression2likelihood[p] for p, count in self.uses.items()) - \
            sum(count * lse([grammar.expression2likelihood[p] for p in ps])
                for ps, count in self.normalizers.items())
    def logLikelihood_overlyGeneral(self, grammar):
        """Calculates log likelihood of this summary, given that the summary might refer to productions that don't occur in the grammar"""
        return self.constant + \
            sum(count * grammar.expression2likelihood[p] for p, count in self.uses.items()) - \
            sum(count * lse([grammar.expression2likelihood.get(p,NEGATIVEINFINITY) for p in ps])
                for ps, count in self.normalizers.items())        
    def numerator(self, grammar):
        return self.constant + \
            sum(count * grammar.expression2likelihood[p] for p, count in self.uses.items())
    def denominator(self, grammar):
        return \
            sum(count * lse([grammar.expression2likelihood[p] for p in ps])
                for ps, count in self.normalizers.items())
    def toUses(self):
        from collections import Counter
        
        possibleVariables = sum( count if Index(0) in ps else 0
                                 for ps, count in self.normalizers.items() )
        actualVariables = self.uses.get(Index(0), 0.)
        actualUses = {k: v
                      for k, v in self.uses.items()
                      if not k.isIndex }
        possibleUses = dict(Counter(p
                                    for ps, count in self.normalizers.items()
                                    for p_ in ps
                                    if not p_.isIndex
                                    for p in [p_]*count ))
        return Uses(possibleVariables, actualVariables,
                    possibleUses, actualUses)


class Uses(object):
    '''Tracks uses of different grammar productions'''

    def __init__(self, possibleVariables=0., actualVariables=0.,
                 possibleUses={}, actualUses={}):
        self.actualVariables = actualVariables
        self.possibleVariables = possibleVariables
        self.possibleUses = possibleUses
        self.actualUses = actualUses

    def __str__(self):
        return "Uses(actualVariables = %f, possibleVariables = %f, actualUses = %s, possibleUses = %s)" %\
            (self.actualVariables, self.possibleVariables, self.actualUses, self.possibleUses)

    def __repr__(self): return str(self)

    def __mul__(self, a):
        return Uses(a * self.possibleVariables,
                    a * self.actualVariables,
                    {p: a * u for p, u in self.possibleUses.items()},
                    {p: a * u for p, u in self.actualUses.items()})

    def __imul__(self, a):
        self.possibleVariables *= a
        self.actualVariables *= a
        for p in self.possibleUses:
            self.possibleUses[p] *= a
        for p in self.actualUses:
            self.actualUses[p] *= a
        return self

    def __rmul__(self, a):
        return self * a

    def __radd__(self, o):
        if o == 0:
            return self
        return self + o

    def __add__(self, o):
        if o == 0:
            return self

        def merge(x, y):
            z = x.copy()
            for k, v in y.items():
                z[k] = v + x.get(k, 0.)
            return z
        return Uses(self.possibleVariables + o.possibleVariables,
                    self.actualVariables + o.actualVariables,
                    merge(self.possibleUses, o.possibleUses),
                    merge(self.actualUses, o.actualUses))

    def __iadd__(self, o):
        self.possibleVariables += o.possibleVariables
        self.actualVariables += o.actualVariables
        for k, v in o.possibleUses.items():
            self.possibleUses[k] = self.possibleUses.get(k, 0.) + v
        for k, v in o.actualUses.items():
            self.actualUses[k] = self.actualUses.get(k, 0.) + v
        return self

    @staticmethod
    def join(z, *weightedUses):
        """Consumes weightedUses"""
        if not weightedUses:
            Uses.empty
        if len(weightedUses) == 1:
            return weightedUses[0][1]
        for w, u in weightedUses:
            u *= exp(w - z)
        total = Uses()
        total.possibleVariables = sum(
            u.possibleVariables for _, u in weightedUses)
        total.actualVariables = sum(u.actualVariables for _, u in weightedUses)
        total.possibleUses = defaultdict(float)
        total.actualUses = defaultdict(float)
        for _, u in weightedUses:
            for k, v in u.possibleUses.items():
                total.possibleUses[k] += v
            for k, v in u.actualUses.items():
                total.actualUses[k] += v
        return total


Uses.empty = Uses()

class ContextualGrammar:
    def __init__(self, noParent, variableParent, library):
        self.noParent, self.variableParent, self.library = noParent, variableParent, library

        self.productions = [(None,t,p) for _,t,p in self.noParent.productions ]
        self.primitives = [p for _,_2,p in self.productions ]

        self.continuationType = noParent.continuationType
        assert variableParent.continuationType == self.continuationType

        assert set(noParent.primitives) == set(variableParent.primitives)
        assert set(variableParent.primitives) == set(library.keys())
        for e,gs in library.items():
            assert len(gs) == len(e.infer().functionArguments())
            for g in gs:
                assert set(g.primitives) == set(library.keys())
                assert g.continuationType == self.continuationType

    def untorch(self):
        return ContextualGrammar(self.noParent.untorch(), self.variableParent.untorch(),
                                 {e: [g.untorch() for g in gs ]
                                  for e,gs in self.library.items() })

    def randomWeights(self, r):
        """returns a new grammar with random weights drawn from r. calls `r` w/ old weight"""
        return ContextualGrammar(self.noParent.randomWeights(r),
                                 self.variableParent.randomWeights(r),
                                 {e: [g.randomWeights(r) for g in gs]
                                  for e,gs in self.library.items() })
    def __str__(self):
        lines = ["No parent:",str(self.noParent),"",
                 "Variable parent:",str(self.variableParent),"",
                 ""]
        for e,gs in self.library.items():
            for j,g in enumerate(gs):
                lines.extend(["Parent %s, argument index %s"%(e,j),
                              str(g),
                              ""])
        return "\n".join(lines)

    def json(self):
        return {"noParent": self.noParent.json(),
                "variableParent": self.variableParent.json(),
                "productions": [{"program": str(e),
                                 "arguments": [gp.json() for gp in gs ]}
                                    for e,gs in self.library.items() ]}

    @staticmethod
    def fromGrammar(g):
        return ContextualGrammar(g, g,
                                 {e: [g]*len(e.infer().functionArguments())
                                  for e in g.primitives })
                

    class LS: # likelihood summary
        def __init__(self, owner):
            self.noParent = LikelihoodSummary()
            self.variableParent = LikelihoodSummary()
            self.library = {e: [LikelihoodSummary() for _ in gs]  for e,gs in owner.library.items() }

        def record(self, parent, parentIndex, actual, possibles, constant):
            if parent is None: ls = self.noParent
            elif parent.isIndex: ls = self.variableParent
            else: ls = self.library[parent][parentIndex]
            ls.record(actual, possibles, constant=constant)

        def join(self, other):
            self.noParent.join(other.noParent)
            self.variableParent.join(other.variableParent)
            for e,gs in self.library.items():
                for g1,g2 in zip(gs, other.library[e]):
                    g1.join(g2)

        def logLikelihood(self, owner):
            return self.noParent.logLikelihood(owner.noParent) + \
                   self.variableParent.logLikelihood(owner.variableParent) + \
                   sum(r.logLikelihood(g)
                       for e, rs in self.library.items()
                       for r,g in zip(rs, owner.library[e]) )            
        def numerator(self, owner):
            return self.noParent.numerator(owner.noParent) + \
                   self.variableParent.numerator(owner.variableParent) + \
                   sum(r.numerator(g)
                       for e, rs in self.library.items()
                       for r,g in zip(rs, owner.library[e]) )            
        def denominator(self, owner):
            return self.noParent.denominator(owner.noParent) + \
                   self.variableParent.denominator(owner.variableParent) + \
                   sum(r.denominator(g)
                       for e, rs in self.library.items()
                       for r,g in zip(rs, owner.library[e]) )            

    def likelihoodSummary(self, parent, parentIndex, context, environment, request, expression):
        if request.isArrow():
            assert expression.isAbstraction
            return self.likelihoodSummary(parent, parentIndex,
                                          context,
                                          [request.arguments[0]] + environment,
                                          request.arguments[1],
                                          expression.body)
        if parent is None: g = self.noParent
        elif parent.isIndex: g = self.variableParent
        else: g = self.library[parent][parentIndex]            
        candidates = g.buildCandidates(request, context, environment,
                                       normalize=False, returnTable=True)

        # A list of everything that would have been possible to use here
        possibles = [p for p in candidates.keys() if not p.isIndex]
        numberOfVariables = sum(p.isIndex for p in candidates.keys())
        if numberOfVariables > 0:
            possibles += [Index(0)]

        f, xs = expression.applicationParse()

        assert f in candidates

        thisSummary = self.LS(self)
        thisSummary.record(parent, parentIndex,
                           f, possibles,
                           constant= -math.log(numberOfVariables) if f.isIndex else 0)

        _, tp, context = candidates[f]
        argumentTypes = tp.functionArguments()
        assert len(xs) == len(argumentTypes)

        for i, (argumentType, argument) in enumerate(zip(argumentTypes, xs)):
            argumentType = argumentType.apply(context)
            context, newSummary = self.likelihoodSummary(f, i,
                                                         context, environment, argumentType, argument)
            thisSummary.join(newSummary)

        return context, thisSummary

    def closedLikelihoodSummary(self, request, expression):
        return self.likelihoodSummary(None,None,
                                      Context.EMPTY,[],
                                      request, expression)[1]

    def logLikelihood(self, request, expression):
        return self.closedLikelihoodSummary(request, expression).logLikelihood(self)

    def sample(self, request, maximumDepth=8, maxAttempts=None):
        attempts = 0
        while True:
            try:
                _, e = self._sample(None, None, Context.EMPTY, [], request, maximumDepth)
                return e
            except NoCandidates:
                if maxAttempts is not None:
                    attempts += 1
                    if attempts > maxAttempts: return None
                continue
            
    def _sample(self, parent, parentIndex, context, environment, request, maximumDepth):
        if request.isArrow():
            context, body = self._sample(parent, parentIndex, context,
                                         [request.arguments[0]] + environment,
                                         request.arguments[1],
                                         maximumDepth)
            return context, Abstraction(body)
        if parent is None: g = self.noParent
        elif parent.isIndex: g = self.variableParent
        else: g = self.library[parent][parentIndex]
        candidates = g.buildCandidates(request, context, environment,
                                       normalize=True, returnProbabilities=True,
                                       mustBeLeaf=(maximumDepth <= 1))
        newType, chosenPrimitive, context = sampleDistribution(candidates)

        xs = newType.functionArguments()
        returnValue = chosenPrimitive

        for j,x in enumerate(xs):
            x = x.apply(context)
            context, x = self._sample(chosenPrimitive, j, context, environment, x, maximumDepth - 1)
            returnValue = Application(returnValue, x)
            
        return context, returnValue

    def expectedUsesMonteCarlo(self, request, debug=None):
        import numpy as np
        n = 0
        u = [0.]*len(self.primitives)
        primitives = list(sorted(self.primitives, key=str))
        noInventions = all( not p.isInvented for p in primitives )
        primitive2index = {primitive: i
                           for i, primitive in enumerate(primitives)
                           if primitive.isInvented or noInventions }
        eprint(primitive2index)
        ns = 10000
        with timing(f"calculated expected uses using Monte Carlo simulation w/ {ns} samples"):
            for _ in range(ns):
                p = self.sample(request, maxAttempts=0)
                if p is None: continue
                n += 1
                if debug and n < 10:
                    eprint(debug, p)
                for _, child in p.walk():
                    if child not in primitive2index: continue
                    u[primitive2index[child]] += 1.0
        u = np.array(u)/n
        if debug:
            eprint(f"Got {n} samples. Feature vector:\n{u}")
            eprint(f"Likely used primitives: {[p for p,i in primitive2index.items() if u[i] > 0.5]}")
            eprint(f"Likely used primitive indices: {[i for p,i in primitive2index.items() if u[i] > 0.5]}")
        return u

    def featureVector(self, _=None, requests=None, onlyInventions=True, normalize=True):
        """
        Returns the probabilities licensed by the type system.
        This is like the grammar productions, but with irrelevant junk removed.
        Its intended use case is for clustering; it should be strictly better than the raw transition matrix.
        """
        if requests is None:
            if self.continuationType: requests = {self.continuationType}
            elif any( 'REAL' == str(p) for p in self.primitives ): requests = set()
            elif any( 'STRING' == str(p) for p in self.primitives ): requests = {tlist(tcharacter)}
            else: requests = set()
        requests = {r.returns() for r in requests}
        features = []
        logWeights = []
        for l,t,p in sorted(self.noParent.productions,
                            key=lambda z: str(z[2])):
            if onlyInventions and not p.isInvented: continue
            if any( canUnify(r, t.returns()) for r in requests ) or len(requests) == 0:
                logWeights.append(l)
        features.append(logWeights)
        for parent in sorted(self.primitives, key=str):
            if onlyInventions and not parent.isInvented: continue
            if parent not in self.library: continue
            argumentTypes = parent.infer().functionArguments()
            for j,g in enumerate(self.library[parent]):
                argumentType = argumentTypes[j]
                logWeights = []
                for l,t,p in sorted(g.productions,
                                    key=lambda z: str(z[2])):
                    if onlyInventions and not p.isInvented: continue
                    if canUnify(argumentType.returns(), t.returns()):
                        logWeights.append(l)
                features.append(logWeights)

        if normalize:
            features = [ [math.exp(w - z) for w in lw ]
                         for lw in features
                         if lw
                         for z in [lse(lw)] ]
        import numpy as np
        return np.array([f
                         for lw in features
                         for f in lw])

    def enumeration(self,context,environment,request,upperBound,
                    parent=None, parentIndex=None,
                    maximumDepth=20,
                    lowerBound=0.):
        '''Enumerates all programs whose MDL satisfies: lowerBound <= MDL < upperBound'''
        if upperBound < 0 or maximumDepth == 1:
            return

        if request.isArrow():
            v = request.arguments[0]
            for l, newContext, b in self.enumeration(context, [v] + environment,
                                                     request.arguments[1],
                                                     parent=parent, parentIndex=parentIndex,
                                                     upperBound=upperBound,
                                                     lowerBound=lowerBound,
                                                     maximumDepth=maximumDepth):
                yield l, newContext, Abstraction(b)
        else:
            if parent is None: g = self.noParent
            elif parent.isIndex: g = self.variableParent
            else: g = self.library[parent][parentIndex]

            candidates = g.buildCandidates(request, context, environment,
                                           normalize=True)

            for l, t, p, newContext in candidates:
                mdl = -l
                if not (mdl < upperBound):
                    continue

                xs = t.functionArguments()
                for aL, aK, application in\
                    self.enumerateApplication(newContext, environment, p, xs,
                                              parent=p,
                                              upperBound=upperBound + l,
                                              lowerBound=lowerBound + l,
                                              maximumDepth=maximumDepth - 1):
                    yield aL + l, aK, application

    def enumerateApplication(self, context, environment,
                             function, argumentRequests,
                             # Upper bound on the description length of all of
                             # the arguments
                             upperBound,
                             # Lower bound on the description length of all of
                             # the arguments
                             lowerBound=0.,
                             maximumDepth=20,
                             parent=None, 
                             originalFunction=None,
                             argumentIndex=0):
        assert parent is not None
        if upperBound < 0. or maximumDepth == 1:
            return
        if originalFunction is None:
            originalFunction = function

        if argumentRequests == []:
            if lowerBound <= 0. and 0. < upperBound:
                yield 0., context, function
            else:
                return
        else:
            argRequest = argumentRequests[0].apply(context)
            laterRequests = argumentRequests[1:]
            for argL, newContext, arg in self.enumeration(context, environment, argRequest,
                                                          parent=parent, parentIndex=argumentIndex,
                                                          upperBound=upperBound,
                                                          lowerBound=0.,
                                                          maximumDepth=maximumDepth):
                if violatesSymmetry(originalFunction, arg, argumentIndex):
                    continue

                newFunction = Application(function, arg)
                for resultL, resultK, result in self.enumerateApplication(newContext, environment, newFunction,
                                                                          laterRequests,
                                                                          parent=parent,
                                                                          upperBound=upperBound + argL,
                                                                          lowerBound=lowerBound + argL,
                                                                          maximumDepth=maximumDepth,
                                                                          originalFunction=originalFunction,
                                                                          argumentIndex=argumentIndex + 1):
                    yield resultL + argL, resultK, result
                
        


def violatesSymmetry(f, x, argumentIndex):
    if not f.isPrimitive:
        return False
    while x.isApplication:
        x = x.f
    if not x.isPrimitive:
        return False
    f = f.name
    x = x.name
    if f == "car":
        return x == "cons" or x == "empty"
    if f == "cdr":
        return x == "cons" or x == "empty"
    if f == "+":
        return x == "0" or (argumentIndex == 1 and x == "+")
    if f == "-":
        return argumentIndex == 1 and x == "0"
    if f == "empty?":
        return x == "cons" or x == "empty"
    if f == "zero?":
        return x == "0" or x == "1"
    if f == "index" or f == "map" or f == "zip":
        return x == "empty"
    if f == "range":
        return x == "0"
    if f == "fold":
        return argumentIndex == 1 and x == "empty"
    return False

def batchLikelihood(jobs):
    """Takes as input a set of (program, request, grammar) and returns a dictionary mapping each of these to its likelihood under the grammar"""
    superGrammar = Grammar.uniform(list({p for _1,_2,g in jobs for p in g.primitives}),
                                   continuationType=list(jobs)[0][-1].continuationType)
    programsAndRequests = {(program, request)
                           for program, request, grammar in jobs}
    with timing(f"Calculated {len(programsAndRequests)} likelihood summaries"):
        summary = {(program, request): superGrammar.closedLikelihoodSummary(request, program)
                   for program, request in programsAndRequests}
    with timing(f"Calculated log likelihoods from summaries"):
        response = {}
        for program, request, grammar in jobs:
            fast = summary[(program, request)].logLikelihood_overlyGeneral(grammar)
            if False: # debugging
                slow = grammar.logLikelihood(request, program)
                print(program)
                eprint(grammar.closedLikelihoodSummary(request, program))
                eprint(superGrammar.closedLikelihoodSummary(request, program))
                print()
                assert abs(fast - slow) < 0.0001
            response[(program, request, grammar)] = fast
    return response

class PCFG():
    def __init__(self, productions, start_symbol, number_of_arguments):
        # productions: nonterminal -> [(log probability, constructor, [(#lambdas, nonterminal)])]
        self.number_of_arguments = number_of_arguments
        self.productions = productions
        self.start_symbol = start_symbol

    
        

    @staticmethod
    def from_grammar(g, request, maximum_type=3, maximum_environment=2):
        kinds = set()

        def process_type(t):
            if t.isArrow():
                process_type(t.arguments[0])
                process_type(t.arguments[1])
            elif isinstance(t, TypeVariable):
                return
            else:
                kinds.add((t.name, len(t.arguments)))
                for a in t.arguments:
                    process_type(a)

        process_type(request)
        for _, t, _ in g.productions:
            process_type(t)


        def types_of_size(s):
            if s<=1:
                yield from { TypeConstructor(n, []) for n, a in kinds if a==0 }
                return 

            for n, a in kinds:
                assert a<3
                if a==0: continue
                if a==1:
                    yield from { TypeConstructor(n, [t]) for t in types_of_size(s-1) }
                if a==2:
                    yield from { TypeConstructor(n, [t1, t2])
                             for s1 in  range(1, s)
                             for s2 in  range(1, s-s1)
                             for t1 in types_of_size(s1)
                             for t2 in types_of_size(s2) }

        def size_of_type(t):
            if isinstance(t, TypeVariable):
                return 0
            if t.isArrow():
                return max(size_of_type(t.arguments[0]),
                           size_of_type(t.arguments[1]))
            return 1 + sum(size_of_type(a) for a in t.arguments)

        environment = tuple(reversed(request.functionArguments()))
        maximum_environment += len(environment)
        request = request.returns()
        possible_types = {t for s in range(1, maximum_type+1) for t in types_of_size(s)}|{request}

        _instantiations = {}
        def instantiations(t):
            if not t.isPolymorphic:
                return [t]

            if t in _instantiations:
                return _instantiations[t]
            
            t=t.canonical()
            variables = t.free_type_variables()
            
            return_value = []
            for substitution in itertools.product(possible_types, repeat=len(variables)):
                context = Context(substitution=list(zip(range(len(variables)), substitution)))
                new_type = t.apply(context)
                if size_of_type(new_type) <= maximum_type:
                    return_value.append(new_type)
            _instantiations[t] = return_value
            return return_value

        # for _, t, p in g.productions:
        #     print(p, t)
        #     for i in instantiations(t):
        #         print("\t", i)

        def push_environment(tp, e):
            if len(e)==0:
                return (tp,)
            else:
                return tuple([tp]+list(e))
            e=dict(e)
            if tp in e: e[tp]+=1
            else: e[tp]=1
            return frozendict(e)
        def push_multiple_environment(ts, e):
            for t in ts: e=push_environment(t, e)
            return e
                

        rules = {}
        def make_rules(request, environment):
            if (request, environment) in rules: return
            rules[(request, environment)] = []
            variable_candidates = [(g.logVariable, tp, Index(i))
                                   # for t, count in environment.items()
                                   # for i in range(count)
                                   for i, t in enumerate(environment) 
                                   for tp  in instantiations(t)
                                   if tp.returns()==request]
            if g.continuationType == request:
                variable_candidates = [min(variable_candidates, key=lambda vc: vc[-1].i)]
            variable_candidates = [(lp-math.log(len(variable_candidates)), t, p)
                                   for lp, t, p in variable_candidates ]
            
            for lp, t, p in g.productions + variable_candidates:
                for i in instantiations(t):
                    if i.returns() == request:
                        arguments = i.functionArguments()
                        argument_symbols = []
                        for a in arguments:
                            new_environment = push_multiple_environment(a.functionArguments(),
                                                                        environment)
                            argument_symbols.append((len(a.functionArguments()),
                                                     (a.returns(), new_environment)))

                        if all( len(new_environment) <= maximum_environment
                                for _, (_, new_environment) in argument_symbols ):
                            rules[(request, environment)].append((lp, p, argument_symbols))
            
            for _, p, argument_symbols in rules[(request, environment)]:
                for _, symbol in argument_symbols:
                    make_rules(*symbol)

        start_environment = push_multiple_environment(environment, {})
        start_symbol = (request, start_environment)
        make_rules(*start_symbol)
        eprint(len(rules), "nonterminal symbols")
        eprint(sum(len(productions) for productions in rules.values()), "production rules")
        return PCFG(rules, start_symbol, len(start_environment)).normalize()

    def normalize(self):
        def norm(distribution):
            z = lse([x[0] for x in distribution])
            return [(x[0]-z, *x[1:]) for x in distribution]
        if isinstance(self.productions, list):
            self.productions = [ norm(rs) for rs in self.productions ]
        elif isinstance(self.productions, dict):
            self.productions = {k:norm(rs) for k, rs in self.productions.items()}
        else:
            assert False
        return self

    def __str__(self):
        return f"start symbol: {self.start_symbol}\n\n%s"%("\n\n".join(
            "\n".join(f"{nt} ::= {k}\t%s\t\t{l}"%(" ".join(f"{n}x{s}" for n, s in ar ))
                      for l, k, ar in rs)
            for nt, rs in self.productions.items()))
            
    def number_rules(self):
        if isinstance(self.productions, list):
            return self
        
        mapping = dict(zip(self.productions.keys(), range(len(self.productions))))
        reverse_mapping = {v:k for k,v in mapping.items() }

        new_productions = [ [ (lp, k, [(nl, mapping[nt]) for nl, nt in arguments ])
                              for lp, k, arguments in self.productions[reverse_mapping[i]] ]
            for i in range(len(self.productions)) ]

        return PCFG(new_productions, mapping[self.start_symbol], self.number_of_arguments)

    
    

    def json(self):
        self = self.number_rules()
        return {"rules": [ [ {"probability": lp, 
                              "constructor": str(k),
                              "arguments": [ {"n_lambda": nl, "nt": nt}
                                             for nl, nt in arguments ]}
                             for (lp, k, arguments) in rules ]
                           for rules in self.productions ],
                "number_of_arguments": self.number_of_arguments,
                "start_symbol": self.start_symbol
        }

    def log_probability(self, program, symbol=None):

        if symbol is None:
            symbol = self.start_symbol

        while isinstance(program, Program) and program.isAbstraction:
            program = program.body

        if isinstance(program, NamedHole):
            program = program.name
            
        if not isinstance(program, Program):
            # assume it is a nonterminal
            assert isinstance(program, int) and 0<=program<len(self.productions) or \
                program in self.productions, f"failure to type production: {program}:{symbol}"

            if program == symbol:
                return 0.
            else:
                return float("-inf")
            
            
        

        rules = self.productions[symbol]

        
        f, xs = program.applicationParse()

        lp = float("-inf")
        for p, k, arguments in rules:
            if f==k:
                _lp=p+sum(self.log_probability(a, at)
                           for a, (_, at) in zip(xs, arguments) )
                lp = lse(lp, _lp)
        return lp

    def best_first_enumeration(self, partial=False):
        h=PQ()

        h.push(0., (0., NamedHole(self.start_symbol).wrap_in_abstractions(self.number_of_arguments)))

        def next_nonterminal(expression):
            if isinstance(expression, NamedHole):
                return expression.name
            
            if expression.isAbstraction:
                return next_nonterminal(expression.body)
            if expression.isApplication:
                f=next_nonterminal(expression.f)
                if f is None:
                    return next_nonterminal(expression.x)
                return f
            return None

        def substitute(expression, value):
            if isinstance(expression, NamedHole):
                return value
            
            if expression.isAbstraction:
                body = substitute(expression.body, value)
                if body is None: return None
                return Abstraction(body)
            if expression.isApplication:
                f = substitute(expression.f, value)
                if f is None:
                    x = substitute(expression.x, value)
                    if x is None: return None
                    return Application(expression.f, x)
                return Application(f, expression.x)
            return None

        

        while len(h)>0:
            lp, e = h.popMaximum()
            
            nt=next_nonterminal(e)
            
            if nt is None:
                yield e, lp
            else:
                for lpp, k, arguments in self.productions[nt]:
                    rewrite = k
                    for nl, at in arguments:
                        at = NamedHole(at).wrap_in_abstractions(nl)
                        rewrite = Application(rewrite, at)
                    #eprint(e, ">>", substitute(e, rewrite))
                    ep = substitute(e, rewrite)
                    h.push(lp+lpp, (lp+lpp, ep))
                    if partial:
                        yield ep, lp+lpp

    def split(self, nc):
        
        
        def expansions(expression):
            if isinstance(expression, NamedHole):
                for _, k, arguments in self.productions[expression.name]:
                    arguments = [NamedHole(at).wrap_in_abstractions(nl)
                                 for nl, at in arguments ]
                    for a in arguments:
                        k = Application(k, a)
                    yield k
            
            if expression.isAbstraction:
                for b in expansions(expression.body):
                    yield Abstraction(b)
            
            if expression.isApplication:
                
                for f in expansions(expression.f):
                    yield Application(f, expression.x)
                for x in expansions(expression.x):
                    yield Application(expression.f, x)

        initial_split = [NamedHole(self.start_symbol).wrap_in_abstractions(self.number_of_arguments)]
        while len(initial_split) < nc:
            biggest=max(initial_split, key=lambda pp: self.log_probability(pp))
            initial_split = list(expansions(biggest)) + [pp for pp in initial_split if pp!=biggest]

        split = [[] for _ in range(nc) ]
        for i, pp in enumerate(initial_split):
            split[i%nc].append(pp)

        return split

        
        # def quality(s):
        #     mass = [ lse([self.log_probability(pp) for pp in pps ])
        #              for pps in s ]
        #     eprint(mass)
        #     return exp(min(mass)-max(mass))

        # def find_swap(s):
        #     i = max(range(s), key=lambda i: lse([self.log_probability(pp) for pp in s[i] ]))
            
        # eprint(quality(split))
        # import pdb; pdb.set_trace()
        

        

        

    def quantized_enumeration(self, resolution=0.5, skeletons=None):        
        self = self.number_rules()

        if skeletons is None:
            skeletons = [NamedHole(self.start_symbol).wrap_in_abstractions(self.number_of_arguments)]
            skeletons = [pp for pps in self.split(10) for pp in pps ]
            eprint(skeletons)
        skeleton_costs=[int(-self.log_probability(pp)/resolution+0.5)
                        for pp in skeletons ]

        # replace probabilities with quantized costs            
        productions = [[ (max(int(-lp/resolution+0.5), 1), k, arguments)
                         for lp, k, arguments in right_hand_sides ]
                       for right_hand_sides in self.productions]
        
        nonterminals = len(productions)

        expressions = [ [None for _ in range(int(100/resolution))]
                        for _ in range(nonterminals) ]

        def expressions_of_size(symbol, size):
            nonlocal expressions
            
            
            if size <= 0:
                return []
            
            if expressions[symbol][size] is None:
                new=[]
                for cost, k, arguments in productions[symbol]:
                    if cost>size: continue
                    
                    if len(arguments) == 0:
                        if cost==size:
                            new.append(k)
                    elif len(arguments) == 1:
                        nl1, at1 = arguments[0]
                        for a1 in expressions_of_size(at1, size-cost):
                            a1 = a1.wrap_in_abstractions(nl1)
                            new.append(Application(k, a1))
                    elif len(arguments) == 2:
                        nl1, at1 = arguments[0]
                        nl2, at2 = arguments[1]
                        for c1 in range(size-cost):
                            for a1 in expressions_of_size(at1, c1):
                                a1 = a1.wrap_in_abstractions(nl1)
                                for a2 in expressions_of_size(at2, size-cost-c1):
                                    a2 = a2.wrap_in_abstractions(nl2)
                                    new.append(Application(Application(k, a1), a2))
                    elif len(arguments) == 3:
                        nl1, at1 = arguments[0]
                        nl2, at2 = arguments[1]
                        nl3, at3 = arguments[2]
                        for c1 in range(size-cost):
                            for a1 in expressions_of_size(at1, c1):
                                a1 = a1.wrap_in_abstractions(nl1)
                                for c2 in range(size-cost-c1):
                                    for a2 in expressions_of_size(at2, c2):
                                        a2 = a2.wrap_in_abstractions(nl2)
                                        for a3 in expressions_of_size(at3, size-cost-c1-c2):
                                            a3 = a3.wrap_in_abstractions(nl3)
                                            new.append(Application(Application(Application(k, a1), a2), a3))
                    elif len(arguments) == 4:
                        nl1, at1 = arguments[0]
                        nl2, at2 = arguments[1]
                        nl3, at3 = arguments[2]
                        nl4, at4 = arguments[3]
                        for c1 in range(size-cost):
                            for a1 in expressions_of_size(at1, c1):
                                a1 = a1.wrap_in_abstractions(nl1)
                                for c2 in range(size-cost-c1):
                                    for a2 in expressions_of_size(at2, c2):
                                        a2 = a2.wrap_in_abstractions(nl2)
                                        for c3 in range(size-cost-c1-c2):
                                            for a3 in expressions_of_size(at3, c3):
                                                a3 = a3.wrap_in_abstractions(nl3)
                                                for a4 in expressions_of_size(at3, size-cost-c1-c2-c3):
                                                    a4 = a4.wrap_in_abstractions(nl4)
                                                    new.append(Application(Application(Application(Application(k, a1), a2), a3), a4))
                    elif len(arguments) == 5:
                        nl1, at1 = arguments[0]
                        nl2, at2 = arguments[1]
                        nl3, at3 = arguments[2]
                        nl4, at4 = arguments[3]
                        nl5, at5 = arguments[4]
                        for c1 in range(size-cost):
                            for a1 in expressions_of_size(at1, c1):
                                a1 = a1.wrap_in_abstractions(nl1)
                                for c2 in range(size-cost-c1):
                                    for a2 in expressions_of_size(at2, c2):
                                        a2 = a2.wrap_in_abstractions(nl2)
                                        for c3 in range(size-cost-c1-c2):
                                            for a3 in expressions_of_size(at3, c3):
                                                a3 = a3.wrap_in_abstractions(nl3)
                                                for c4 in range(size-cost-c1-c2-c3):
                                                    for a4 in expressions_of_size(at3, c4):
                                                        a4 = a4.wrap_in_abstractions(nl4)
                                                        for a5 in expressions_of_size(at3, size-cost-c1-c2-c3-c4):
                                                            a5 = a4.wrap_in_abstractions(nl5)
                                                            new.append(Application(Application(Application(Application(Application(k, a1), a2), a3), a4), a5))
                    else:
                        assert False, "more than five arguments not supported for the enumeration algorithm but that is not for any good reason"
                expressions[symbol][size] = new
                
            return expressions[symbol][size]

        def complete_skeleton(cost, skeleton):
            if skeleton.isAbstraction:
                for b in complete_skeleton(cost, skeleton.body):
                    yield Abstraction(b)
            elif skeleton.isApplication:
                for function_cost in range(cost+1):
                    for f in complete_skeleton(function_cost, skeleton.f):
                        for x in complete_skeleton(cost-function_cost, skeleton.x):
                            yield Application(f, x)
            elif skeleton.isNamedHole:
                yield from expressions_of_size(skeleton.name, cost)
            else:
                if cost==0:
                    yield skeleton

                    
        

        expressions = [ [None for _ in range(int(100/resolution))]
                        for _ in range(nonterminals) ]
        for cost in range(int(100/resolution)):
            for skeleton, skeleton_cost in zip(skeletons, skeleton_costs):
                for e in complete_skeleton(cost-skeleton_cost, skeleton):                    
                    yield e

        

    
