from utilities import *

from fragmentUtilities import *
from frontier import *
from grammar import *
from program import *

from digamma import *
from itertools import izip
import gc


class FragmentGrammar(object):
    def __init__(self, logVariable, productions):
        self.logVariable = logVariable
        self.productions = productions
        self.likelihoodCache = {}

    def clearCache(self):
        self.likelihoodCache = {}

    def __repr__(self):
        return "FragmentGrammar(logVariable={self.logVariable}, productions={self.productions}".format(self=self)
    def __str__(self):
        def productionKey((l,t,p)):
            return not isinstance(p,Primitive), -l
        return "\n".join(["%f\tt0\t$_"%self.logVariable] + \
                         [ "%f\t%s\t%s"%(l,t,p) for l,t,p in sorted(self.productions, key = productionKey) ])
                                                                    

    def buildCandidates(self, context, environment, request):
        candidates = []
        variableCandidates = []
        for l,t,p in self.productions:
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
                variableCandidates.append((newContext,
                                           t.apply(newContext),
                                           Index(j)))
            except UnificationFailure: continue
        if variableCandidates:
            z = math.log(len(variableCandidates))
            for newContext, newType, index in variableCandidates:
                candidates.append((self.logVariable - z, newContext, newType, index))

        z = lse([candidate[0] for candidate in candidates])
        return [(l - z, c, t, p) for l, c, t, p in candidates]

    def closedLogLikelihood(self, request, expression):
        _,l,_ = self.logLikelihood(Context.EMPTY, [], request, expression)
        return l

    def closedUses(self, request, expression):
        _,l,u = self.logLikelihood(Context.EMPTY, [], request, expression)
        return l,u

    def logLikelihood(self, context, environment, request, expression):
        '''returns (context, log likelihood, uses)'''

        # We can cash likelihood calculations faster whenever they don't involve type inference
        # This is because they are guaranteed to not modify the context, 
        polymorphic = request.isPolymorphic or any(v.isPolymorphic for v in environment)
        # For some reason polymorphic caching slows it down
        shouldDoCaching = not polymorphic

        # Caching
        if shouldDoCaching:
            if polymorphic:
                inTypes = canonicalTypes([request.apply(context)] + [ v.apply(context) for v in environment])
            else:
                inTypes = canonicalTypes([request] + environment)
            cacheKey = (tuple(inTypes), expression)
            if cacheKey in self.likelihoodCache:
                outTypes, l, u = self.likelihoodCache[cacheKey]
                context, instantiatedTypes = instantiateTypes(context, outTypes)
                outRequest = instantiatedTypes[0]
                outEnvironment = instantiatedTypes[1:]
                # eprint("request:", request.apply(context), "environment:",
                #        [ v.apply(context) for v in environment ])
                # eprint("will be unified with: out request:",outRequest,"out environment",outEnvironment)
                if polymorphic:
                    context = context.unify(request, outRequest)
                    for v,vp in zip(environment, outEnvironment):
                        context = context.unify(v, vp)                
                return context,l,u            
        
        if request.isArrow():
            if not isinstance(expression,Abstraction): return (context,NEGATIVEINFINITY,Uses.empty)
            return self.logLikelihood(context,
                                      [request.arguments[0]] + environment,
                                      request.arguments[1],
                                      expression.body)
        
        # Not a function type

        # Construct and normalize the candidate productions
        candidates = self.buildCandidates(context, environment, request)

        # Consider each way of breaking the expression up into a
        # function and arguments
        totalLikelihood = NEGATIVEINFINITY
        weightedUses = []

        possibleVariables = float(int(any(isinstance(candidate,Index)
                                          for _,_,_,candidate in candidates )))
        possibleUses = {candidate: 1. for _,_,_,candidate in candidates
                                      if not isinstance(candidate,Index)}
        
        for f,xs in expression.applicationParses():
            for candidateLikelihood, newContext, tp, production in candidates:
                variableBindings = {}
                # This is a variable in the environment
                if production.isIndex:
                    if production != f: continue
                else:
                    try:
                        newContext, fragmentType, variableBindings = \
                                            Matcher.match(newContext, production, f, len(xs))
                        # This is necessary because the types of the variable
                        # bindings and holes need to match up w/ request
                        fragmentTypeTemplate = request
                        for _ in xs:
                            newContext, newVariable = newContext.makeVariable()
                            fragmentTypeTemplate = arrow(newVariable, fragmentTypeTemplate)
                        newContext = newContext.unify(fragmentType, fragmentTypeTemplate)
                        # update the unified type
                        tp = fragmentType.apply(newContext)
                    except MatchFailure: continue

                argumentTypes = tp.functionArguments()
                if len(xs) != len(argumentTypes):
                    # I think that this is some kind of bug. But I can't figure it out right now.
                    # As a hack, count this as though it were a failure
                    continue
                    #raise GrammarFailure('len(xs) != len(argumentTypes): tp={}, xs={}'.format(tp, xs))


                thisLikelihood = candidateLikelihood
                if isinstance(production, Index):
                    theseUses = Uses(possibleVariables=possibleVariables,
                                     actualVariables=1.,
                                     possibleUses=possibleUses.copy(),
                                     actualUses={})
                else:
                    theseUses = Uses(possibleVariables=possibleVariables,
                                     actualVariables=0.,
                                     possibleUses=possibleUses.copy(),
                                     actualUses={production: 1.})

                # Accumulate likelihood from free variables and holes and arguments
                for freeType,freeExpression in variableBindings.values() + zip(argumentTypes, xs):
                    freeType = freeType.apply(newContext)
                    newContext, expressionLikelihood, newUses = \
                            self.logLikelihood(newContext, environment, freeType, freeExpression)
                    if expressionLikelihood is NEGATIVEINFINITY:
                        thisLikelihood = NEGATIVEINFINITY
                        break
                    
                    thisLikelihood += expressionLikelihood
                    theseUses += newUses

                if thisLikelihood is NEGATIVEINFINITY: continue

                weightedUses.append((thisLikelihood,theseUses))
                totalLikelihood = lse(totalLikelihood, thisLikelihood)

                # Any of these new context objects should be equally good
                context = newContext

        if totalLikelihood is NEGATIVEINFINITY: return context, totalLikelihood, Uses.empty
        assert weightedUses != []

        allUses = Uses.join(totalLikelihood, *weightedUses)

        # memoize result
        if shouldDoCaching:
            outTypes = [ request.apply(context) ] + [ v.apply(context) for v in environment ]
            outTypes = canonicalTypes(outTypes)
            self.likelihoodCache[cacheKey] = (outTypes, totalLikelihood, allUses)

        return context, totalLikelihood, allUses

    def expectedUses(self, frontiers):
        likelihoods = [ [ (l + entry.logLikelihood, u)
                         for entry in frontier
                         for l,u in [self.closedUses(frontier.task.request, entry.program)] ]
                        for frontier in frontiers ]
        zs = (lse([ l for l,_ in ls ]) for ls in likelihoods)
        return sum(math.exp(l - z)*u
                   for z,frontier in izip(zs,likelihoods)
                   for l,u in frontier)

    def insideOutside(self, frontiers, pseudoCounts):
        uses = self.expectedUses(frontiers)
        return FragmentGrammar(log(uses.actualVariables + pseudoCounts)
                               - log(uses.possibleVariables),
                               [ (log(uses.actualUses.get(p,0.) + pseudoCounts)
                                  - log(uses.possibleUses.get(p,pseudoCounts)),
                                  t,p)
                                 for _,t,p in self.productions ])

    def variationalUpdate(self, frontiers, uses):
        realUses = self.expectedUses(frontiers)
        uses = realUses + uses
        def f(actual, possible):
            if possible == 0:
                assert actual == 0
                return 0.
            return digamma(actual) - digamma(possible)
        return FragmentGrammar(f(uses.actualVariables, uses.possibleVariables),
                               [ (f(uses.actualUses[p], uses.possibleUses[p]), t, p)
                                 for _,t,p in self.productions
                                 if realUses.actualUses.get(p,0) > 1 or
                                    isinstance(p,(Primitive,Invented))
                               ]).normalize()

    def jointFrontiersLikelihood(self, frontiers):
        return sum( lse([ entry.logLikelihood + self.closedLogLikelihood(frontier.task.request, entry.program)
                          for entry in frontier ])
                    for frontier in frontiers )
    def jointFrontiersMDL(self, frontiers):
        return sum( max( entry.logLikelihood + self.closedLogLikelihood(frontier.task.request, entry.program)
                          for entry in frontier )
                    for frontier in frontiers )

    def __len__(self): return len(self.productions)

    @staticmethod
    def fromGrammar(g):
        return FragmentGrammar(g.logVariable, g.productions)
    def toGrammar(self):
        return Grammar(self.logVariable, [(l,q.infer(),q)
                                          for l,t,p in self.productions
                                          for q in [defragment(p)] ])

    @property
    def primitives(self): return [ p for _,_,p in self.productions ]

    @staticmethod
    def uniform(productions):
        return FragmentGrammar(0., [(0., p.infer(),p) for p in productions ])
    def normalize(self):
        z = lse([ l for l,t,p in self.productions ] + [self.logVariable])
        return FragmentGrammar(self.logVariable - z,
                               [ (l - z,t,p) for l,t,p in self.productions ])
    def makeUniform(self):
        return FragmentGrammar(0., [(0., p.infer(),p) for _,_,p in self.productions ])

    def rescoreFrontier(self, frontier):
        return Frontier([ FrontierEntry(e.program,
                                        logPrior = self.closedLogLikelihood(frontier.task.request, e.program),
                                        logLikelihood = e.logLikelihood)
                          for e in frontier ],
                        frontier.task)

    @staticmethod
    def induceVariational(g0, frontiers, _ = None,
                          topK = 1, pseudoCounts = 1.0, aic = 1.0, structurePenalty = 0.001, a = 0, CPUs = 1):
        grammar = FragmentGrammar.fromGrammar(g0)
        frontiers = [frontier for frontier in frontiers if not frontier.empty ]
        eprint("Inducing a grammar from",len(frontiers),"frontiers using variational inference")

        # "restricted frontiers" only contain the top K according to the best grammar
        def restrictFrontiers():
            return parallelMap(CPUs, lambda f: grammar.rescoreFrontier(f).topK(topK),
                               frontiers)
        restrictedFrontiers = restrictFrontiers()
        fragments = [ fragment for fragment in proposeFragmentsFromFrontiers(restrictedFrontiers, a)
                      if not fragment in grammar.primitives ]

        def priorCounts(p):
            size = fragmentSize(p) + 1.
            epsilon = 0.1
            pc = log(1./(1. - exp(-size)))/log(1./epsilon)
            return pc

        # Add all the fragments to the grammar
        grammar = FragmentGrammar(log(priorCounts(Index(0))),
                                  [ (log(priorCounts(p)), p.infer(), p)
                                    for p in grammar.primitives + fragments ])
        # prior
        uses0 = Uses(actualVariables = priorCounts(Index(0)),
                     possibleVariables = 0,
                     actualUses = {p: priorCounts(p) for p in grammar.primitives },
                     possibleUses = {p: 0 for p in grammar.primitives })
        
        for i in xrange(5):
            eprint("VB iteration",i)
            grammar = grammar.variationalUpdate(restrictedFrontiers, uses0).normalize()
            restrictedFrontiers = restrictFrontiers()

            grammar.productions.sort(key = lambda (l,_,__): -l)
            eprint(grammar)

        uses = grammar.expectedUses(frontiers)
        retainedPrimitives = [ p for p in grammar.primitives
                               if uses.actualUses[p] > 2 or isinstance(p,(Primitive,Invented)) ]
        grammar = FragmentGrammar.uniform(retainedPrimitives).insideOutside(frontiers, pseudoCounts)
        return grammar

                
    @staticmethod
    def induceFromFrontiers(g0, frontiers, _ = None,
                            topK = 1, pseudoCounts = 1.0, aic = 1.0, structurePenalty = 0.001, a = 0, CPUs = 1):
        frontiers = [frontier for frontier in frontiers if not frontier.empty ]
        eprint("Inducing a grammar from",len(frontiers),"frontiers")
        
        bestGrammar = FragmentGrammar.fromGrammar(g0)

        # "restricted frontiers" only contain the top K according to the best grammar
        def restrictFrontiers():
            return parallelMap(CPUs, lambda f: bestGrammar.rescoreFrontier(f).topK(topK),
                               frontiers)
        restrictedFrontiers = []

        def grammarScore(g):
            g = g.makeUniform().insideOutside(restrictedFrontiers, pseudoCounts)
            likelihood = g.jointFrontiersMDL(restrictedFrontiers)
            structure = sum(fragmentSize(p) for p in g.primitives)
            score = likelihood - aic*len(g) - structurePenalty*structure
            g.clearCache()
            gc.collect()
            return score, g


        if aic is not POSITIVEINFINITY:
            restrictedFrontiers = restrictFrontiers()
            bestScore, _ = grammarScore(bestGrammar)
            while True:
                restrictedFrontiers = restrictFrontiers()
                fragments = [ fragment for fragment in proposeFragmentsFromFrontiers(restrictedFrontiers, a)
                              if not fragment in bestGrammar.primitives ]
                eprint("Proposed %d fragments."%len(fragments))

                candidateGrammars = [ FragmentGrammar.uniform(bestGrammar.primitives + [fragment])
                                      for fragment in fragments ]
                if not candidateGrammars:
                    break

                scoredFragments = parallelMap(CPUs, grammarScore, candidateGrammars)
                newScore, newGrammar = max(scoredFragments)

                if newScore <= bestScore:
                    break
                dS = newScore - bestScore
                bestScore, bestGrammar = newScore, newGrammar
                newPrimitiveLikelihood,newType,newPrimitive = bestGrammar.productions[-1]
                eprint("New primitive of type %s\t%s (score = %f; dScore = %f)"%(newType,newPrimitive,newScore,dS))
                # Rewrite the frontiers in terms of the new fragment
                if False:
                    concretePrimitive = defragment(newPrimitive)
                    bestGrammar.productions[-1] = (newPrimitiveLikelihood,
                                                   concretePrimitive.tp,
                                                   concretePrimitive)
                    frontiers = parallelMap(CPUs,
                                            lambda frontier: RewriteFragments.rewriteFrontier(frontier, newPrimitive),
                                            frontiers)
        else:
            eprint("Skipping fragment proposals")

        if False:
            # Reestimate the parameters using the entire frontiers
            bestGrammar = bestGrammar.makeUniform().insideOutside(frontiers, pseudoCounts)
        elif True:
            # Reestimate the parameters using the best programs
            restrictedFrontiers = restrictFrontiers()
            bestGrammar = bestGrammar.makeUniform().insideOutside(restrictedFrontiers, pseudoCounts)
        else:
            # Use parameters that were found during search
            pass
            

        eprint("Old joint = %f\tNew joint = %f\n"%(FragmentGrammar.fromGrammar(g0).jointFrontiersMDL(frontiers),
                                                   bestGrammar.jointFrontiersMDL(frontiers)))
        bestGrammar.clearCache()
        return bestGrammar

def induceFragmentGrammarFromFrontiers(*arguments, **keywordArguments):
    with timing("Induced a grammar"):
        g = FragmentGrammar.induceFromFrontiers(*arguments, **keywordArguments)
        # Experimental variational inference does not seem to work well...
        #g = FragmentGrammar.induceVariational(*arguments, **keywordArguments)
    return g
