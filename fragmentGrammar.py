from utilities import *

from fragmentUtilities import *
from frontier import *
from grammar import *
from program import *

import time
from digamma import *
        

class FragmentGrammar(object):
    def __init__(self, logVariable, productions):
        self.logVariable = logVariable
        self.productions = productions

    def __repr__(self):
        return "FragmentGrammar(logVariable={self.logVariable}, productions={self.productions}".format(self=self)
    def __str__(self):
        def productionKey((l,t,p)):
            return not isinstance(p,Primitive), -l
        return "\n".join(["%f\tt0\t$_"%self.logVariable] + \
                         [ "%f\t%s\t%s"%(l,t,p) for l,t,p in sorted(self.productions, key = productionKey) ])
                                                                    

    def buildCandidates(self, context, environment, request):
        candidates = []
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
                candidates.append((self.logVariable,newContext,
                                   t.apply(newContext),
                                   Index(j)))
            except UnificationFailure: continue
        
        z = lse([candidate[0] for candidate in candidates])
        return [(l - z, k, c, p) for l,k,c,p in candidates ]

    def closedLogLikelihood(self, request, expression):
        _,l,_ = self.logLikelihood(Context.EMPTY, [], request, expression)
        return l

    def logLikelihood(self, context, environment, request, expression):
        '''returns (context, log likelihood, uses)'''
        #eprint("REQUEST",request,"EXPRESSION",expression)
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
        
        for f,xs in expression.applicationParses():
            for candidateLikelihood, newContext, tp, production in candidates:
                variableBindings = {}
                # This is a variable in the environment
                if isinstance(production, Index):
                    if production != f: continue
                else:
                    try:
                        # eprint("Trying to match %s w/ %s"%(production, f))
                        newContext, fragmentType, variableBindings = \
                                            Matcher.match(newContext, production, f, len(xs))
                        # This is necessary because the types of the variable
                        # bindings and holes need to match up w/ request
                        # eprint("Fragment type",fragmentType)
                        fragmentTypeTemplate = request
                        for _ in xs:
                            newContext, newVariable = newContext.makeVariable()
                            fragmentTypeTemplate = arrow(newVariable, fragmentTypeTemplate)
                        newContext = newContext.unify(fragmentType, fragmentTypeTemplate)
                        # eprint("Fragment type after unification w/ template",fragmentType.apply(newContext))
                        # eprint("H = ",[(t.apply(newContext),h) for t,h in holes ],\)
                        #     "V = ",{i: (t.apply(newContext),v) for i,(t,v) in variableBindings.iteritems() }
                        # update the unified type
                        tp = fragmentType.apply(newContext)
                    except MatchFailure: continue

                thisLikelihood = candidateLikelihood
                theseUses = Uses(possibleVariables = float(int(any(isinstance(candidate,Index)
                                                                   for _,_,_,candidate in candidates ))),
                                 actualVariables = float(int(isinstance(production,Index))),
                                 possibleUses = {candidate: 1.
                                                 for _,_,_,candidate in candidates
                                                 if not isinstance(candidate,Index)},
                                 actualUses = {} if isinstance(production,Index) else {production: 1.})

                # eprint("tp",tp)
                # eprint("tp.functionArguments",tp.functionArguments())
                # eprint("xs = ",xs)
                argumentTypes = tp.functionArguments()
                assert len(xs) == len(argumentTypes)

                # Accumulate likelihood from free variables and holes and arguments
                for freeType,freeExpression in variableBindings.values() + zip(argumentTypes, xs):
                    freeType = freeType.apply(newContext)
                    newContext, expressionLikelihood, newUses = \
                            self.logLikelihood(newContext, environment, freeType, freeExpression)
                    if expressionLikelihood is NEGATIVEINFINITY:
                        thisLikelihood = NEGATIVEINFINITY
                        break
                    
                    thisLikelihood += expressionLikelihood
                    theseUses = theseUses + newUses

                if thisLikelihood is NEGATIVEINFINITY: continue

                weightedUses.append((thisLikelihood,theseUses))
                totalLikelihood = lse(totalLikelihood, thisLikelihood)

                # Any of these new context objects should be equally good
                context = newContext

        if totalLikelihood is NEGATIVEINFINITY: return context, totalLikelihood, Uses.empty
        assert weightedUses != []

        allUses = Uses.empty
        for w,u in weightedUses:
            allUses = allUses + (u*exp(w - totalLikelihood))

        return context, totalLikelihood, allUses

    def expectedUses(self, frontiers):
        likelihoods = [ [ (l + entry.logLikelihood, u) 
                          for entry in frontier
                          for _,l,u in [self.logLikelihood(Context.EMPTY, [], frontier.task.request, entry.program)] ]
                        for frontier in frontiers ]
        zs = [ lse([ l for l,_ in ls ])
               for ls in likelihoods ]
        return sum([ math.exp(l - z)*u
                     for z,frontier in zip(zs,likelihoods)
                     for l,u in frontier ])

    def insideOutside(self, frontiers, pseudoCounts):
        uses = self.expectedUses(frontiers)
        return FragmentGrammar(log(uses.actualVariables + pseudoCounts) - \
                               log(uses.possibleVariables + pseudoCounts),
                               [ (log(uses.actualUses.get(p,0.) + pseudoCounts) - \
                                  log(uses.possibleUses.get(p,0.) + pseudoCounts),
                                  t,p)
                                 for _,t,p in self.productions ])

    def variationalUpdate(self, frontiers, uses):
        uses = self.expectedUses(frontiers) + uses
        def f(actual, possible): return digamma(actual) - digamma(possible)
        return FragmentGrammar(f(uses.actualVariables, uses.possibleVariables),
                               [ (f(uses.actualUses[p], uses.possibleUses[p]), t, p)
                                 for _,t,p in self.productions ]).normalize()

    def jointFrontiersLikelihood(self, frontiers):
        return sum( lse([ entry.logLikelihood + self.closedLogLikelihood(frontier.task.request, entry.program)
                          for entry in frontier ])
                    for frontier in frontiers )
    def jointFrontiersMDL(self, frontiers):
        return sum( max([ entry.logLikelihood + self.closedLogLikelihood(frontier.task.request, entry.program)
                          for entry in frontier ])
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
                     possibleVariables = 1.,
                     actualUses = {p: priorCounts(p) for p in grammar.primitives },
                     possibleUses = {p: 1. for p in grammar.primitives })
        
        for i in range(2):
            eprint("VB iteration",i)
            grammar = grammar.variationalUpdate(restrictedFrontiers, uses0)
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

        initialLogPrior = [  [ bestGrammar.closedLogLikelihood(f.task.request, e.program) for e in f ]
                             for f in frontiers ]
            
        
        # "restricted frontiers" only contain the top K according to the best grammar
        def restrictFrontiers():
            return parallelMap(CPUs, lambda f: bestGrammar.rescoreFrontier(f).topK(topK),
                               frontiers)
        restrictedFrontiers = restrictFrontiers()

        def grammarScore(g):
            newFragment = str(g.primitive[-1])
            g = g.makeUniform().insideOutside(restrictedFrontiers, pseudoCounts)
            likelihood = g.jointFrontiersMDL(restrictedFrontiers)
            structure = sum(fragmentSize(p) for p in g.primitives)
            score = likelihood - aic*len(g) - structurePenalty*structure, g

            if "join" in newFragment and "split" in newFragment and "map" in newFragment:
                eprint("Interesting new fragment %s obtains likelihood %f, aic %f, structure penalty %f, score %f"%\
                       (newFragment, likelihood, aic*len(g), structurePenalty*structure, score))
            return score

        bestScore, _ = grammarScore(bestGrammar)

        while True:
            restrictedFrontiers = restrictFrontiers()
            fragments = [ fragment for fragment in proposeFragmentsFromFrontiers(restrictedFrontiers, a)
                          if not fragment in bestGrammar.primitives ]
                
            candidateGrammars = [ FragmentGrammar.uniform(bestGrammar.primitives + [fragment])
                                  for fragment in fragments ]
            if candidateGrammars == []: break

            scoredFragments = parallelMap(CPUs, grammarScore, candidateGrammars)
            (newScore, newGrammar) = max(scoredFragments)
                    
            if newScore > bestScore:
                dS = newScore - bestScore
                bestScore, bestGrammar = newScore, newGrammar
                _,newType,newPrimitive = bestGrammar.productions[-1]
                eprint("New primitive of type %s\t%s (score = %f; dScore = %f)"%(newType,newPrimitive,newScore,dS))
            else: break

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
        return bestGrammar

def induceFragmentGrammarFromFrontiers(*arguments, **keywordArguments):
    startTime = time.time()
    g = FragmentGrammar.induceFromFrontiers(*arguments, **keywordArguments)
    # Experimental variational inference does not seem to work well...
    #g = FragmentGrammar.induceVariational(*arguments, **keywordArguments)
    eprint("Grammar induction took time",time.time() - startTime,"seconds")
    return g


    
