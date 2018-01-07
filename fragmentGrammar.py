
from frontier import *
from grammar import *
from program import *

class FragmentVariable(Program):
    def __init__(self): pass
    def __str__(self): return "??"
    def __eq__(self,o): return isinstance(o,FragmentVariable)
    def __hash__(self): return 42
    def evaluate(self, e):
        raise Exception('Attempt to evaluate fragment variable')
    def inferType(self,context, environment):
        return context.makeVariable()
    def shift(self,offset,depth = 0):
        raise Exception('Attempt to shift fragment variable')
    def match(self,expression,variableBindings,surroundingAbstractions = 0):
        try:
            return [expression.shift(-surroundingAbstractions)]
        except ShiftFailure: raise MatchFailure()

FragmentVariable.single = FragmentVariable()

def proposeFragmentsFromProgram(p,a):

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
            for f in fragment(expression.f,a): yield f
            for f in fragment(expression.x,a): yield f
        elif isinstance(expression, Abstraction):
            for f in fragment(expression.body,a): yield f
        else:
            assert isinstance(expression, (Invented,Primitive,Index))

    return { f for b in range(a + 1) for f in fragments(p,b) }

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

    def logLikelihood(self, context, environment, request, expression):
        '''returns (context, type, log likelihood)'''
        if request.isArrow():
            if not isinstance(expression,Abstraction): return (None,None,NEGATIVEINFINITY)
            (context,bodyType,l) = \
                self.logLikelihood(context,
                                   [request.arguments[0]] + environment,
                                   request.arguments[1],
                                   expression.body)
            return (context,arrow(request.arguments[0],bodyType).apply(context),l)
        
        # Not a function type

        # Consider each way of breaking the expression up into a
        # function and arguments
        for f,xs in expression.applicationParses():
            pass
            
        
                                      
            

if __name__ == "__main__":
    addition = Primitive("+",
                     arrow(tint,arrow(tint,tint)),
                     lambda x: lambda y: x + y)
    k0 = Primitive("0",tint,0)
    expression = Abstraction(Application(Application(addition,k0),Index(0)))
    for f in proposeFragmentsFromProgram(expression,1):
        try:
            variables = {}
            bindings = f.match(expression, variables)
            print "Matched expression %s with fragment %s"%(expression,f)
            print bindings
            print variables
        except MatchFailure:
            print "Cannot match expression %s with fragment %s"%(expression,f)
        print 

    
        

