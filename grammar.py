from collections import defaultdict
from itertools import izip

from frontier import *
from program import *
from type import *

class GrammarFailure(Exception): pass

class Grammar(object):
    def __init__(self, logVariable, productions):
        self.logVariable = logVariable
        self.productions = productions

        self.expression2likelihood = dict( (p,l) for l,_,p in productions)
        self.expression2likelihood[Index(0)] = self.logVariable

    @staticmethod
    def fromProductions(productions, logVariable=0.0):
        """Make a grammar from primitives and their relative logpriors."""
        return Grammar(logVariable, [(l, p.infer(), p) for l, p in productions])

    @staticmethod
    def uniform(primitives):
        return Grammar(0.0, [(0.0,p.infer(),p) for p in primitives ])

    def __len__(self): return len(self.productions)
    def __str__(self):
        def productionKey((l,t,p)):
            return not isinstance(p,Primitive), -l
        lines = ["%f\tt0\t$_"%self.logVariable]
        for l,t,p in sorted(self.productions, key = productionKey):
            l = "%f\t%s\t%s"%(l,t,p)
            if not t.isArrow() and isinstance(p,Invented):
                l += "\teval = %s"%(p.evaluate([]))
            lines.append(l)
        return "\n".join(lines)

    @property
    def primitives(self):
        return [p for _, _, p in self.productions]

    def buildCandidates(self, request, context, environment,
                        # Should the log probabilities be normalized?
                        normalize = True,
                        # Should be returned a table mapping primitives to their candidate entry?
                        returnTable = False,
                        # Should we return probabilities vs log probabilities?
                        returnProbabilities = False):
        """Your all-in-one stop for building data structures representing the primitives with a certain type
        If returnTable is false (default): returns [((log)likelihood, tp, primitive, context)]
        if returntable is true: returns {primitive: ((log)likelihood, tp, context)}"""
        if returnProbabilities: assert normalize
        
        candidates = []
        variableCandidates = []
        for l,t,p in self.productions:
            try:
                newContext, t = t.instantiate(context)
                newContext = newContext.unify(t.returns(), request)
                candidates.append((l,t.apply(newContext),p,newContext))
            except UnificationFailure: continue
        for j,t in enumerate(environment):
            try:
                newContext = context.unify(t.returns(), request)
                variableCandidates.append((t.apply(newContext), Index(j), newContext))
            except UnificationFailure: continue

        candidates = [ (self.logVariable - log(len(variableCandidates)), t, p, k)
                       for t,p,k in variableCandidates ] + candidates
        if normalize:
            z = lse([ l for l,t,p,k in candidates ])
            if returnProbabilities:
                candidates = [ (exp(l - z), t, p, k) for l,t,p,k in candidates ]
            else: candidates = [ (l - z, t, p, k) for l,t,p,k in candidates ]


        if returnTable:
            return {p: (l,t,k) for l,t,p,k in candidates }
        else:
            return candidates

    def sample(self, request):
        _,e = self._sample(request, Context.EMPTY, [])
        return e
    def _sample(self, request, context, environment):
        if request.isArrow():
            return self._sample(request.arguments[1],
                               context,
                               [request.arguments[0]] + environment)
        candidates = self.buildCandidates(requests, context, environment,
                                          normalize = True,
                                          returnProbabilities = True)
        newType, chosenPrimitive, context = sampleDistribution(candidates)

        # Sample the arguments
        xs = newType.functionArguments()
        returnValue = chosenPrimitive

        for x in xs:
            x = x.apply(context)
            context, x = self._sample(x, context, environment)
            returnValue = Application(returnValue, x)

        return context, returnValue        
        

    def likelihoodSummary(self, context, environment, request, expression):
        if request.isArrow():
            if not isinstance(expression,Abstraction): return context,None
            return self.likelihoodSummary(context,
                                          [request.arguments[0]] + environment,
                                          request.arguments[1],
                                          expression.body)
        # Build the candidates
        candidates = self.buildCandidates(request, context, environment,
                                          normalize = False,
                                          returnTable = True)
        
        # A list of everything that would have been possible to use here
        possibles = [ p for p in candidates.keys() if not p.isIndex ]
        numberOfVariables = sum(p.isIndex for p in candidates.keys())
        if numberOfVariables > 0: possibles += [Index(0)]

        f,xs = expression.applicationParse()

        if f not in candidates: return context,None

        thisSummary = LikelihoodSummary()
        thisSummary.record(f, possibles,
                           constant = -math.log(numberOfVariables) if f.isIndex else 0)

        _, context, tp = candidates[f]
        argumentTypes = tp.functionArguments()
        assert len(xs) == len(argumentTypes)

        for argumentType, argument in zip(argumentTypes, xs):
            argumentType = argumentType.apply(context)
            context, newSummary = self.likelihoodSummary(context, environment, argumentType, argument)
            if newSummary is None: return context, None
            thisSummary.join(newSummary)

        return context, thisSummary

    def closedLikelihoodSummary(self, request, expression):
        context, summary = self.likelihoodSummary(Context.EMPTY, [], request, expression)
        return summary

    def closedLogLikelihood(self, request, expression):
        summary = self.closedLikelihoodSummary(request, expression)
        return summary.logLikelihood(self)

    def rescoreFrontier(self, frontier):
        return Frontier([ FrontierEntry(e.program,
                                        logPrior = self.closedLogLikelihood(frontier.task.request, e.program),
                                        logLikelihood = e.logLikelihood)
                          for e in frontier ],
                        frontier.task)
        

class LikelihoodSummary(object):
    '''Summarizes the terms that will be used in a likelihood calculation'''
    def __init__(self):
        self.uses = {}
        self.normalizers = {}
        self.constant = 0.
    def __str__(self):
        return """LikelihoodSummary(constant = %f, 
uses = {%s},
normalizers = {%s})"""%(self.constant,
                        ", ".join("%s: %d"%(k,v) for k,v in self.uses.iteritems() ),
                        ", ".join("%s: %d"%(k,v) for k,v in self.normalizers.iteritems() ))
    def record(self, actual, possibles, constant = 0.):
        # Variables are all normalized to be $0
        if isinstance(actual, Index): actual = Index(0)

        # Make it something that we can hash
        possibles = frozenset(sorted(possibles))
        
        self.constant += constant
        self.uses[actual] = self.uses.get(actual,0) + 1
        self.normalizers[possibles]  = self.normalizers.get(possibles,0) + 1
    def join(self, other):
        self.constant += other.constant
        for k,v in other.uses.iteritems(): self.uses[k] = self.uses.get(k,0) + v
        for k,v in other.normalizers.iteritems(): self.normalizers[k] = self.normalizers.get(k,0) + v
    def logLikelihood(self, grammar):
        return self.constant + \
            sum(count * grammar.expression2likelihood[p] for p, count in self.uses.iteritems() ) - \
            sum(count * lse([grammar.expression2likelihood[p] for p in ps ])
                for ps, count in self.normalizers.iteritems() )
            
            

        
class Uses(object):
    '''Tracks uses of different grammar productions'''
    def __init__(self, possibleVariables = 0., actualVariables = 0.,
                 possibleUses = {}, actualUses = {}):
        self.actualVariables = actualVariables
        self.possibleVariables = possibleVariables
        self.possibleUses = possibleUses
        self.actualUses = actualUses

    def __str__(self):
        return "Uses(actualVariables = %f, possibleVariables = %f, actualUses = %s, possibleUses = %s)"%\
            (self.actualVariables, self.possibleVariables, self.actualUses, self.possibleUses)
    def __repr__(self): return str(self)

    def __mul__(self,a):
        return Uses(a*self.possibleVariables,
                    a*self.actualVariables,
                    {p: a*u for p,u in self.possibleUses.iteritems() },
                    {p: a*u for p,u in self.actualUses.iteritems() })
    def __imul__(self,a):
        self.possibleVariables *= a
        self.actualVariables *= a
        for p in self.possibleUses:
            self.possibleUses[p] *= a
        for p in self.actualUses:
            self.actualUses[p] *= a
        return self
    def __rmul__(self,a):
        return self*a
    def __radd__(self,o):
        if o == 0: return self
        return self + o
    def __add__(self,o):
        if o == 0: return self
        def merge(x,y):
            z = x.copy()
            for k,v in y.iteritems():
                z[k] = v + x.get(k,0.)
            return z
        return Uses(self.possibleVariables + o.possibleVariables,
                    self.actualVariables + o.actualVariables,
                    merge(self.possibleUses,o.possibleUses),
                    merge(self.actualUses,o.actualUses))
    def __iadd__(self,o):
        self.possibleVariables += o.possibleVariables
        self.actualVariables += o.actualVariables
        for k, v in o.possibleUses.iteritems():
            self.possibleUses[k] = self.possibleUses.get(k, 0.) + v
        for k, v in o.actualUses.iteritems():
            self.actualUses[k] = self.actualUses.get(k, 0.) + v
        return self

    @staticmethod
    def join(z, *weightedUses):
        """Consumes weightedUses"""
        if not weightedUses: Uses.empty
        for w, u in weightedUses:
            u *= exp(w - z)
        total = Uses()
        total.possibleVariables = sum(u.possibleVariables for _, u in weightedUses)
        total.actualVariables = sum(u.actualVariables for _, u in weightedUses)
        total.possibleUses = defaultdict(float)
        total.actualUses = defaultdict(float)
        for _, u in weightedUses:
            for k, v in u.possibleUses.iteritems():
                total.possibleUses[k] += v
            for k, v in u.actualUses.iteritems():
                total.actualUses[k] += v
        return total
    
Uses.empty = Uses()
