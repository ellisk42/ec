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

class FragmentGrammar():
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
        
        z = math.log(sum(math.exp(candidate[0]) for candidate in candidates))
        return [(l - z, k, c, p) for l,k,c,p in candidates ]

    def closedLogLikelihood(self, request, expression):
        #print "About to correctly the likelihood of",expression
        _,l = self.logLikelihood(Context.EMPTY, [], request, expression)
        #print "Got likelihood",l
        return l

    def logLikelihood(self, context, environment, request, expression):
        '''returns (context, log likelihood)'''
        #print "REQUEST",request,"EXPRESSION",expression
        if request.isArrow():
            if not isinstance(expression,Abstraction): return (context,NEGATIVEINFINITY)
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

                # print "tp",tp
                # print "tp.functionArguments",tp.functionArguments()
                # print "xs = ",xs
                argumentTypes = tp.functionArguments()
                assert len(xs) == len(argumentTypes)

                # Accumulate likelihood from free variables and holes and arguments
                for freeType,freeExpression in variableBindings.values() + holes + zip(argumentTypes, xs):
                    freeType = freeType.apply(newContext)
                    newContext, expressionLikelihood = \
                            self.logLikelihood(newContext, environment, freeType, freeExpression)
                    thisLikelihood += expressionLikelihood
                    if thisLikelihood == NEGATIVEINFINITY: break

                if thisLikelihood == NEGATIVEINFINITY: continue

                totalLikelihood = lse(totalLikelihood, thisLikelihood)

                # Any of these new context objects should be equally good
                context = newContext

        return context, totalLikelihood

    def __len__(self): return len(self.productions)

    @staticmethod
    def fromGrammar(g):
        return FragmentGrammar(g.logVariable, g.productions)
    @staticmethod
    def uniform(productions):
        return FragmentGrammar(0., [(0., p.infer(),p) for p in productions ])
                
    @staticmethod
    def induceFromFrontiers(g0, frontiers, pseudoCounts = 1.0, aic = 1.0, structurePenalty = 0.5, a = 1):
        frontiers = [frontier for frontier in frontiers if not frontier.empty() ]
        fragments = proposeFragmentsFromFrontiers(frontiers,a)

        for f in fragments: print f

        def grammarScore(productions):
            g = FragmentGrammar.uniform(productions)
            likelihood = sum( lse([ entry.logLikelihood + \
                                    g.closedLogLikelihood(frontier.task.request, entry.program)
                                    for entry in frontier ])
                              for frontier in frontiers )
            structure = sum(p.size() for p in productions)
            return likelihood - aic*len(g) - structurePenalty*structure

        bestProductions = [p for _,_,p in g0.productions ]
        bestScore = grammarScore(bestProductions)

        while True:
            newScore = None
            newProductions = None
            
            for f in fragments:
                if f in bestProductions: continue
                thisScore = grammarScore(bestProductions + [f])
                if newScore == None or thisScore > newScore:
                    newScore = thisScore
                    newProductions = bestProductions + [f]
            if newScore > bestScore:
                bestScore = newScore
                bestProductions = newProductions
                print "Updated grammar to: (score = %f)"%newScore
                print FragmentGrammar.uniform(bestProductions)
                print
            else: break
        
        
        
