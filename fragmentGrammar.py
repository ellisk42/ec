from utilities import *

from frontier import *
from grammar import *
from program import *

class FragmentVariable(Program):
    def __init__(self): pass
    def show(self,isFunction): return "??"
    def __eq__(self,o): return isinstance(o,FragmentVariable)
    def __hash__(self): return 42
    def evaluate(self, e):
        raise Exception('Attempt to evaluate fragment variable')
    def inferType(self,context, environment, freeVariables):
        return context.makeVariable()
    def shift(self,offset,depth = 0):
        raise Exception('Attempt to shift fragment variable')
    def match(self, context, expression, holes, variableBindings, environment = []):
        surroundingAbstractions = len(environment)
        try:
            context, variable = context.makeVariable()
            holes.append((variable, expression.shift(-surroundingAbstractions)))
            return context, variable
        except ShiftFailure: raise MatchFailure()

    def walk(self, surroundingAbstractions = 0): yield surroundingAbstractions,self

    def size(self): return 1

FragmentVariable.single = FragmentVariable()

def defragment(expression):
    if isinstance(expression, (Primitive,Invented)): return expression
    
    numberOfAbstractions = [0]
    mapping = {}
    def f(e,d):
        if isinstance(e,FragmentVariable):
            numberOfAbstractions[0] += 1
            return Index(numberOfAbstractions[0] + d - 1)
        if isinstance(e,(Primitive,Invented)): return e
        if isinstance(e,Application):
            return Application(f(e.f,d), f(e.x,d))
        if isinstance(e,Abstraction): return Abstraction(f(e.body),d + 1)
        if isinstance(e,Index):
            if e.i < d: return e
            i = e.i - d
            if i in mapping: return Index(d + mapping[i])
            mapping[i] = numberOfAbstractions[0]
            numberOfAbstractions[0] += 1
            return Index(numberOfAbstractions[0] - 1 + d)
        assert False
    expression = f(expression,0)
    for _ in range(numberOfAbstractions[0]):
        expression = Abstraction(expression)
    return Invented(expression)
            

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
            if isinstance(child,Index) and child.i >= surroundingAbstractions: numberOfVariables += 1
        #print "Fragment %s has %d calls and %d variables and %d primitives"%(f,numberOfHoles,numberOfVariables,numberOfPrimitives)

        return numberOfPrimitives + 0.5 * (numberOfHoles + numberOfVariables) > 1.5            

    return { f for b in range(arity + 1) for f in fragments(p,b) if nontrivial(f) }

def proposeFragmentsFromFrontiers(frontiers,a):
    fragmentsFromEachFrontier = [ { f
                                    for entry in frontier.entries
                                    for f in proposeFragmentsFromProgram(entry.program,a) }
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
        #print "About to correctly the likelihood of",expression
        _,l,_ = self.logLikelihood(Context.EMPTY, [], request, expression)
        #print "Got likelihood",l
        return l

    def logLikelihood(self, context, environment, request, expression):
        '''returns (context, log likelihood, uses)'''
        #print "REQUEST",request,"EXPRESSION",expression
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
                holes = []
                # This is a variable in the environment
                if isinstance(production, Index):
                    if production != f: continue
                else:
                    try:
                        # print "Trying to match %s w/ %s"%(production, f)
                        newContext, fragmentType = production.match(newContext, f, holes, variableBindings)
                        # This is necessary because the types of the variable
                        # bindings and holes need to match up w/ request
                        # print "Fragment type",fragmentType
                        fragmentTypeTemplate = request
                        for _ in xs:
                            newContext, newVariable = newContext.makeVariable()
                            fragmentTypeTemplate = arrow(newVariable, fragmentTypeTemplate)
                        newContext = newContext.unify(fragmentType, fragmentTypeTemplate)
                        # print "Fragment type after unification w/ template",fragmentType.apply(newContext)
                        # print "H = ",[(t.apply(newContext),h) for t,h in holes ],\
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

                # print "tp",tp
                # print "tp.functionArguments",tp.functionArguments()
                # print "xs = ",xs
                argumentTypes = tp.functionArguments()
                assert len(xs) == len(argumentTypes)

                # Accumulate likelihood from free variables and holes and arguments
                for freeType,freeExpression in variableBindings.values() + holes + zip(argumentTypes, xs):
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
        print "Inducing a grammar from",len(frontiers),"frontiers"
        
        bestGrammar = FragmentGrammar.fromGrammar(g0)
        
        # "restricted frontiers" only contain the top K according to the best grammar
        def restrictFrontiers():
            return [ bestGrammar.rescoreFrontier(f).topK(topK) for f in frontiers ]
        restrictedFrontiers = restrictFrontiers()

        def grammarScore(g):
            g = g.makeUniform().insideOutside(restrictedFrontiers, pseudoCounts)
            likelihood = g.jointFrontiersMDL(restrictedFrontiers)
            structure = sum(p.size() for p in g.primitives)
            score = likelihood - aic*len(g) - structurePenalty*structure, g
            return score

        bestScore, _ = grammarScore(bestGrammar)

        while True:
            restrictedFrontiers = restrictFrontiers()
            fragments = proposeFragmentsFromFrontiers(restrictedFrontiers, a)
            
            candidateGrammars = [ FragmentGrammar.uniform(bestGrammar.primitives + [fragment])
                                  for fragment in fragments
                                  if not fragment in bestGrammar.primitives ]
            if candidateGrammars == []: break

            scoredFragments = parallelMap(CPUs, grammarScore, candidateGrammars)
            (newScore, newGrammar) = max(scoredFragments)
            
            if newScore > bestScore:
                bestScore, bestGrammar = newScore, newGrammar                
                print "Updated grammar to: (score = %f)"%newScore
                print newGrammar
                print
            else: break

        # Reestimate the parameters using the entire frontiers
        return bestGrammar.makeUniform().insideOutside(frontiers, pseudoCounts)
        
