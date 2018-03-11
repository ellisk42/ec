from type import *
from program import *
from frontier import *

from collections import Counter

class MatchFailure(Exception): pass
class Matcher(object):
    def __init__(self, context):
        self.context = context
        self.variableBindings = {}

    @staticmethod
    def match(context, fragment, expression, numberOfArguments):
        if not mightMatch(fragment, expression): raise MatchFailure()
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
            for j in xrange(numberOfArguments): expression = Application(expression, Index(j))
            for _ in xrange(numberOfArguments): expression = Abstraction(expression)

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

def mightMatch(f,e,d = 0):
    '''Checks whether fragment f might be able to match against expression e'''
    if f.isIndex:
        if f.bound(d): return f == e
        return True
    if f.isPrimitive or f.isInvented: return f == e
    if f.isAbstraction: return e.isAbstraction and mightMatch(f.body, e.body, d + 1)
    if f.isApplication:
        return e.isApplication and mightMatch(f.x,e.x,d) and mightMatch(f.f,e.f,d)
    assert False

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

def fragmentSize(f, boundVariableCost = 0.1, freeVariableCost = 0.01):
    freeVariables = 0
    leaves = 0
    boundVariables = 0
    for surroundingAbstractions,e in f.walk():
        if isinstance(e,(Primitive,Invented)): leaves += 1
        if isinstance(e,Index):
            if e.bound(surroundingAbstractions): boundVariables += 1
            else: freeVariables += 1
        assert not isinstance(e,FragmentVariable)
    return leaves + boundVariableCost*boundVariables + freeVariableCost*freeVariables

def defragment(expression):
    '''Converts a fragment into an invented primitive'''
    if isinstance(expression, (Primitive,Invented)): return expression

    expression = canonicalFragment(expression)

    for _ in xrange(expression.numberOfFreeVariables):
        expression = Abstraction(expression)

    return Invented(expression)

class RewriteFragments(object):
    def __init__(self, fragment):
        self.fragment = fragment
        self.concrete = defragment(fragment)

    def tryRewrite(self, e, numberOfArguments):
        try:
            context, t, bindings = Matcher.match(Context.EMPTY, self.fragment, e, numberOfArguments)
        except MatchFailure: return None
        
        assert frozenset(bindings.keys()) == frozenset(range(len(bindings))),\
            "Perhaps the fragment is not in canonical form?"
        e = self.concrete
        for j in range(len(bindings)-1, -1, -1):
            _,b = bindings[j]
            e = Application(e, b)
        return e
            
    def application(self, e, numberOfArguments):
        e = Application(e.f.visit(self, numberOfArguments+1),
                        e.x.visit(self, 0))
        return self.tryRewrite(e, numberOfArguments) or e
    def index(self, e, numberOfArguments): return e
    def invented(self, e, numberOfArguments): return e
    def primitive(self, e, numberOfArguments): return e
    def abstraction(self, e, numberOfArguments):
        e = Abstraction(e.body.visit(self, 0))
        return self.tryRewrite(e, numberOfArguments) or e

    def rewrite(self,e): return e.visit(self,0)

    @staticmethod
    def rewriteFrontier(frontier, fragment):
        worker = RewriteFragments(fragment)
        return Frontier([ FrontierEntry(program = worker.rewrite(e.program),
                                        logLikelihood = e.logLikelihood,
                                        logPrior = e.logPrior,
                                        logPosterior = e.logPosterior)
                          for e in frontier ],
                        task = frontier.task)
        

def proposeFragmentsFromFragment(f):
    '''Abstracts out repeated structure within a single fragment'''
    yield f
    freeVariables = f.numberOfFreeVariables
    closedSubtrees = Counter(subtree for _, subtree in f.walk()
                             if not isinstance(subtree, Index) and subtree.closed)
    del closedSubtrees[f]
    for subtree, freq in closedSubtrees.iteritems():
        if freq < 2: continue
        yield canonicalFragment(f.substitute(subtree, Index(freeVariables)))

def nontrivial(f):
    if not isinstance(f, Application): return False
    # Curry
    if isinstance(f.x, FragmentVariable): return False
    if isinstance(f.x, Index):
        # Make sure that the index is used somewhere else
        if not any(isinstance(child, Index) and child.i - surroundingAbstractions == f.x.i
                   for surroundingAbstractions, child in f.f.walk()):
            return False

    numberOfHoles = 0
    numberOfVariables = 0
    numberOfPrimitives = 0
    for surroundingAbstractions,child in f.walk():
        if isinstance(child,(Primitive,Invented)): numberOfPrimitives += 1
        if isinstance(child,FragmentVariable): numberOfHoles += 1
        if isinstance(child,Index) and child.free(surroundingAbstractions): numberOfVariables += 1
    #eprint("Fragment %s has %d calls and %d variables and %d primitives"%(f,numberOfHoles,numberOfVariables,numberOfPrimitives))

    return numberOfPrimitives + 0.5 * (numberOfHoles + numberOfVariables) > 1.5 and numberOfPrimitives >= 1

def proposeFragmentsFromProgram(p,arity):

    def fragment(expression,a):
        """Generates fragments that unify with expression"""
        
        if a == 1:
            yield FragmentVariable.single
        if a == 0:
            yield expression
            return

        if isinstance(expression, Abstraction):
            # Symmetry breaking: (lambda x. f(x)) defragments to be the same as f(x)
            #for b in fragment(expression.body,a): yield Abstraction(b)
            pass
        elif isinstance(expression, Application):
            for fa in xrange(a + 1):
                for f in fragment(expression.f,fa):
                    for x in fragment(expression.x,a - fa):
                        yield Application(f,x)
        else:
            assert isinstance(expression, (Invented,Primitive,Index))

    def fragments(expression,a):
        """Generates fragments that unify with subexpressions of expression"""
        
        for f in fragment(expression,a): yield f
        if isinstance(expression, Application):
            curry = True
            if curry:
                for f in fragments(expression.f, a): yield f
                for f in fragments(expression.x, a): yield f
            else:
                # Pretend that it is not curried
                function,arguments = expression.applicationParse()
                for f in fragments(function,a): yield f
                for argument in arguments:
                    for f in fragments(argument,a): yield f
        elif isinstance(expression, Abstraction):
            for f in fragments(expression.body,a): yield f
        else:
            assert isinstance(expression, (Invented,Primitive,Index))

    return { canonicalFragment(f) for b in xrange(arity + 1) for f in fragments(p,b) if nontrivial(f) }

def proposeFragmentsFromFrontiers(frontiers, a):
    fragmentsFromEachFrontier = [ { fp
                                    for entry in frontier.entries
                                    for f in proposeFragmentsFromProgram(entry.program,a)
                                    for fp in proposeFragmentsFromFragment(f) }
                                  for frontier in frontiers ]
    allFragments = Counter(f for frontierFragments in fragmentsFromEachFrontier
                           for f in frontierFragments)
    return [ fragment for fragment, frequency in allFragments.iteritems()
             if frequency >= 2 and fragment.wellTyped() and nontrivial(fragment) ]
