from utilities import *

from frontier import *
from grammar import *
from program import *

import time

class MatchFailure(Exception): pass
class Matcher(object):
    def __init__(self, context):
        self.context = context
        self.variableBindings = {}

    @staticmethod
    def match(context, fragment, expression, numberOfArguments):
        m = Matcher(context)
        tp = fragment.visit(m, expression, [], numberOfArguments)
        return m.context, tp, m.variableBindings        
    
    def application(self, fragment, expression, environment, numberOfArguments):
        '''returns tp of fragment.'''
        if not isinstance(expression,Application): raise MatchFailure()
        
        ft = fragment.f.visit(self, expression.f, environment, numberOfArguments + 1)
        xt = fragment.x.visit(self, expression.x, environment, 0)
        
        self.context, returnType = self.context.makeVariable()
        try: self.context = self.context.unify(ft,arrow(xt,returnType))
        except UnificationFailure: raise MatchFailure()
        
        return returnType.apply(self.context)
    def index(self, fragment, expression, environment, numberOfArguments):
        # This is a bound variable
        surroundingAbstractions = len(environment)
        if fragment.bound(surroundingAbstractions):
            if expression == fragment:
                return environment[fragment.i].apply(self.context)
            else: raise MatchFailure()
        
        # This is a free variable
        i = fragment.i - surroundingAbstractions
        # The value is going to be lifted out of the fragment. Make
        # sure that it doesn't refer to anything bound by a lambda in
        # the fragment.
        try: expression = expression.shift(-surroundingAbstractions)
        except ShiftFailure: raise MatchFailure()

        # Wrap it in the appropriate number of lambda expressions & applications
        # This is because everything has to be in eta-longform
        if numberOfArguments > 0:
            expression = expression.shift(numberOfArguments)
            for j in range(numberOfArguments): expression = Application(expression, Index(j))
            for _ in range(numberOfArguments): expression = Abstraction(expression)

        # Added to the bindings
        if i in self.variableBindings:
            (tp,binding) = self.variableBindings[i]
            if binding != expression: raise MatchFailure()
        else:
            self.context, tp = self.context.makeVariable()
            self.variableBindings[i] = (tp,expression)
        return tp
    def abstraction(self, fragment, expression, environment, numberOfArguments):
        if not isinstance(expression, Abstraction): raise MatchFailure()

        self.context,argumentType = self.context.makeVariable()
        returnType = fragment.body.visit(self, expression.body, [argumentType] + environment, 0)

        return arrow(argumentType,returnType)
    def primitive(self, fragment, expression, environment, numberOfArguments):
        if fragment != expression: raise MatchFailure()
        self.context,tp = fragment.tp.instantiate(self.context)
        return tp
    def invented(self, fragment, expression, environment, numberOfArguments):
        if fragment != expression: raise MatchFailure()
        self.context,tp = fragment.tp.instantiate(self.context)
        return tp
    def fragmentVariable(self, fragment, expression, environment, numberOfArguments):
        raise Exception('Deprecated: matching against fragment variables. Convert fragment to canonical form to get rid of fragment variables.')
    


def canonicalFragment(expression):
    '''
    Puts a fragment into a canonical form:
    1. removes all FragmentVariable's
    2. renames all free variables based on depth first traversal
    '''
    return expression.visit(CanonicalVisitor(),0)
class CanonicalVisitor(object):
    def __init__(self):
        self.numberOfAbstractions = 0
        self.mapping = {}
    def fragmentVariable(self,e, d):
        self.numberOfAbstractions += 1
        return Index(self.numberOfAbstractions + d - 1)
    def primitive(self,e,d): return e
    def invented(self,e,d): return e
    def application(self,e,d):
        return Application(e.f.visit(self,d), e.x.visit(self,d))
    def abstraction(self,e,d):
        return Abstraction(e.body.visit(self,d + 1))
    def index(self,e,d):
        if e.bound(d): return e
        i = e.i - d
        if i in self.mapping: return Index(d + self.mapping[i])
        self.mapping[i] = self.numberOfAbstractions
        self.numberOfAbstractions += 1
        return Index(self.numberOfAbstractions - 1 + d)

def fragmentSize(f, variableCost = 0.3):
    freeVariables = 0
    leaves = 0
    for surroundingAbstractions,e in f.walk():
        if isinstance(e,(Primitive,Invented)): leaves += 1
        if isinstance(e,Index):
            if surroundingAbstractions > e.i: leaves += 1
            else: freeVariables += 1
        assert not isinstance(e,FragmentVariable)
    return leaves + variableCost*freeVariables
        
def defragment(expression):
    '''Converts a fragment into an invented primitive'''
    if isinstance(expression, (Primitive,Invented)): return expression

    expression = canonicalFragment(expression)
    
    for _ in range(expression.numberOfFreeVariables):
        expression = Abstraction(expression)
    
    return Invented(expression)

def proposeFragmentsFromFragment(f):
    '''Abstracts out repeated structure within a single fragment'''
    yield f
    freeVariables = f.numberOfFreeVariables
    closedSubtrees = [ subtree for _,subtree in f.walk()
                       if subtree != f and not isinstance(subtree, Index) and subtree.closed ]
    for subtree in set(closedSubtrees):
        frequency = sum(t == subtree for t in closedSubtrees )
        if frequency < 2: continue
        fp = canonicalFragment(f.substitute(subtree, Index(freeVariables)))
        #eprint("Fragment from fragment:",f,"\t",fp)
        yield fp

def proposeFragmentsFromProgram(p,arity):

    def fragment(expression,a):
        if a == 1:
            yield FragmentVariable.single
        if a == 0:
            yield expression
            return 

        if isinstance(expression, Abstraction):
            for b in fragment(expression.body,a): yield Abstraction(b)
        elif isinstance(expression, Application):
            for fa in range(a + 1):
                for f in fragment(expression.f,fa):
                    for x in fragment(expression.x,a - fa):
                        yield Application(f,x)
        else:
            assert isinstance(expression, (Invented,Primitive,Index))

    def fragments(expression,a):
        for f in fragment(expression,a): yield f
        if isinstance(expression, Application):
            for f in fragments(expression.f,a): yield f
            for f in fragments(expression.x,a): yield f
        elif isinstance(expression, Abstraction):
            for f in fragments(expression.body,a): yield f
        else:
            assert isinstance(expression, (Invented,Primitive,Index))

    def nontrivial(f):
        if not isinstance(f, Application): return False
        # Curry
        if isinstance(f.x,FragmentVariable): return False
        if isinstance(f.x, Index):
            # Make sure that the index is used somewhere else
            usedElsewhere = False
            for surroundingAbstractions,child in f.f.walk():
                if isinstance(child, Index) and child.i - surroundingAbstractions == f.x.i:
                    usedElsewhere = True
                    break
            if not usedElsewhere: return False

        numberOfHoles = 0
        numberOfVariables = 0
        numberOfPrimitives = 0
        for surroundingAbstractions,child in f.walk():
            if isinstance(child,(Primitive,Invented)): numberOfPrimitives += 1
            if isinstance(child,FragmentVariable): numberOfHoles += 1
            if isinstance(child,Index) and child.free(surroundingAbstractions): numberOfVariables += 1
        #eprint("Fragment %s has %d calls and %d variables and %d primitives"%(f,numberOfHoles,numberOfVariables,numberOfPrimitives))

        return numberOfPrimitives + 0.5 * (numberOfHoles + numberOfVariables) > 1.5            

    return { canonicalFragment(f) for b in range(arity + 1) for f in fragments(p,b) if nontrivial(f) }

def proposeFragmentsFromFrontiers(frontiers,a):
    fragmentsFromEachFrontier = [ { fp
                                    for entry in frontier.entries
                                    for f in proposeFragmentsFromProgram(entry.program,a)
                                    for fp in proposeFragmentsFromFragment(f) }
                                  for frontier in frontiers ]
    allFragments = { f for frontierFragments in fragmentsFromEachFrontier for f in frontierFragments }
    frequencies = {}
    for f in allFragments:
        frequencies[f] = 0
        for frontierFragments in fragmentsFromEachFrontier:
            if f in frontierFragments: frequencies[f] += 1
    return [ fragment for fragment,frequency in frequencies.iteritems() if frequency >= 2 ]

class Uses(object):
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
        for k,v in o.possibleUses:
            self.possibleUses[k] = self.possibleUses.get(k,0.) + v
        for k,v in o.actualUses:
            self.actualUses[k] = self.actualUses.get(k,0.) + v
        return self
    
Uses.empty = Uses()
        

class FragmentGrammar(object):
    def __init__(self, logVariable, productions):
        self.logVariable = logVariable
        self.productions = productions

    def __repr__(self):
        return "FragmentGrammar(logVariable={self.logVariable}, productions={self.productions}".format(self=self)
    def __str__(self):
        return "\n".join(["%f\tt0\t$_"%self.logVariable] + [ "%f\t%s\t%s"%(l,t,p) for l,t,p in self.productions ])

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

    def insideOutside(self, frontiers, pseudoCounts):
        likelihoods = [ [ (l + entry.logLikelihood, u) 
                          for entry in frontier
                          for _,l,u in [self.logLikelihood(Context.EMPTY, [], frontier.task.request, entry.program)] ]
                        for frontier in frontiers ]
        zs = [ lse([ l for l,_ in ls ])
               for ls in likelihoods ]
        uses = sum([ math.exp(l - z)*u
                     for z,frontier in zip(zs,likelihoods)
                     for l,u in frontier ])
        return FragmentGrammar(log(uses.actualVariables + pseudoCounts) - \
                               log(uses.possibleVariables + pseudoCounts),
                               [ (log(uses.actualUses.get(p,0.) + pseudoCounts) - \
                                  log(uses.possibleUses.get(p,0.) + pseudoCounts),
                                  t,p)
                                 for _,t,p in self.productions ])

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
    def makeUniform(self):
        return FragmentGrammar(0., [(0., p.infer(),p) for _,_,p in self.productions ])

    def rescoreFrontier(self, frontier):
        return Frontier([ FrontierEntry(e.program,
                                        logPrior = self.closedLogLikelihood(frontier.task.request, e.program),
                                        logLikelihood = e.logLikelihood)
                          for e in frontier ],
                        frontier.task)
                
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
            g = g.makeUniform().insideOutside(restrictedFrontiers, pseudoCounts)
            likelihood = g.jointFrontiersMDL(restrictedFrontiers)
            structure = sum(fragmentSize(p) for p in g.primitives)
            score = likelihood - aic*len(g) - structurePenalty*structure, g
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
    eprint("Grammar induction took time",time.time() - startTime,"seconds")
    return g


    
