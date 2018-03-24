# -*- coding: utf-8 -*-

from type import *
from utilities import *

from time import time
import math


class InferenceFailure(Exception): pass
class ShiftFailure(Exception): pass
class ParseFailure(Exception): pass

class Program(object):
    def __repr__(self): return str(self)
    def __ne__(self,o): return not (self == o)
    def __str__(self): return self.show(False)
    def canHaveType(self, t):
        try:
            context, actualType = self.inferType(Context.EMPTY,[],{})
            context, t = t.instantiate(context)
            context.unify(t, actualType)
            return True
        except UnificationFailure as e: return False
    def infer(self):
        try:
            return self.inferType(Context.EMPTY,[],{})[1].canonical()
        except UnificationFailure as e:
            raise InferenceFailure(self, e)
    def uncurry(self):
        t = self.infer()
        a = len(t.functionArguments())
        e = self
        existingAbstractions = 0
        while e.isAbstraction:
            e = e.body
            existingAbstractions += 1
        newAbstractions = a - existingAbstractions
        assert newAbstractions >= 0

        # e is the body stripped of abstractions

        for n in reversed(xrange(newAbstractions)):
            e = Application(e, Index(n+existingAbstractions))
        for _ in xrange(a): e = Abstraction(e)

        eprint("Curry",self,e)
        eprint(self.infer())
        eprint(e.infer())
        assert self.infer() == e.infer()
        return e
    def wellTyped(self):
        try:
            self.infer()
            return True
        except InferenceFailure:
            return False
    def runWithArguments(self, xs):
        f = self.evaluate([])
        for x in xs: f = f(x)
        return f
    def applicationParses(self): yield self,[]
    def applicationParse(self): return self,[]
    @property
    def closed(self):
        for surroundingAbstractions, child in self.walk():
            if isinstance(child, FragmentVariable): return False
            if isinstance(child, Index) and child.free(surroundingAbstractions): return False
        return True
    @property
    def numberOfFreeVariables(expression):
        n = 0
        for surroundingAbstractions, child in expression.walk():
            # Free variable
            if isinstance(child, Index) and child.free(surroundingAbstractions):
                n = max(n, child.i - surroundingAbstractions + 1)
        return n

    @property
    def isIndex(self): return False
    @property
    def isApplication(self): return False
    @property
    def isAbstraction(self): return False
    @property
    def isPrimitive(self): return False
    @property
    def isInvented(self): return False

    @staticmethod
    def parse(s):
        e,s = Program._parse(s.strip())
        if s != "": raise ParseFailure(s)
        return e
    @staticmethod
    def _parse(s):
        while len(s) > 0 and s[0].isspace(): s = s[1:]
        for p in [Application, Abstraction, Index, Invented, FragmentVariable, Primitive]:
            try: return p._parse(s)
            except ParseFailure: continue
        raise ParseFailure(s)

class Application(Program):
    '''Function application'''
    def __init__(self,f,x):
        self.f = f
        self.x = x
        self.hashCode = None
        self.isConditional = f.isApplication and \
                             f.f.isApplication and \
                             f.f.f.isPrimitive and \
                             f.f.f.name == "if"
        if self.isConditional:
            self.falseBranch = x
            self.trueBranch = f.x
            self.branch = f.f.x
            
    @property
    def isApplication(self): return True
    def __eq__(self,other): return isinstance(other,Application) and self.f == other.f and self.x == other.x
    def __hash__(self):
        if self.hashCode == None:
            self.hashCode = hash((hash(self.f), hash(self.x)))
        return self.hashCode
    def visit(self, visitor, *arguments, **keywords): return visitor.application(self, *arguments, **keywords)
    def show(self, isFunction):
        if isFunction: return "%s %s"%(self.f.show(True), self.x.show(False))
        else: return "(%s %s)"%(self.f.show(True), self.x.show(False))
    def evaluate(self,environment):
        if self.isConditional:
            if self.branch.evaluate(environment):
                return self.trueBranch.evaluate(environment)
            else:
                return self.falseBranch.evaluate(environment)
        else:
            return self.f.evaluate(environment)(self.x.evaluate(environment))
    def inferType(self,context,environment,freeVariables):
        (context,ft) = self.f.inferType(context,environment,freeVariables)
        (context,xt) = self.x.inferType(context,environment,freeVariables)
        (context,returnType) = context.makeVariable()
        context = context.unify(ft,arrow(xt,returnType))
        return (context, returnType.apply(context))

    def applicationParses(self):
        yield self,[]
        for f,xs in self.f.applicationParses():
            yield f,xs + [self.x]
    def applicationParse(self):
        f,xs = self.f.applicationParse()
        return f,xs + [self.x]

    def shift(self, offset, depth = 0):
        return Application(self.f.shift(offset, depth),
                           self.x.shift(offset, depth))
    def substitute(self, old, new):
        if self == old: return new
        return Application(self.f.substitute(old, new), self.x.substitute(old, new))

    def walkUncurried(self, d = 0):
        yield d,self
        f,xs = self.applicationParse()
        for k in f.walkUncurried(d): yield k
        for x in xs:
            for k in x.walkUncurried(d): yield k

    def walk(self,surroundingAbstractions = 0):
        yield surroundingAbstractions,self
        for child in self.f.walk(surroundingAbstractions): yield child
        for child in self.x.walk(surroundingAbstractions): yield child

    def size(self): return self.f.size() + self.x.size()

    @staticmethod
    def _parse(s):
        while len(s) > 0 and s[0].isspace(): s = s[1:]
        if s == "" or s[0] != '(': raise ParseFailure(s)
        s = s[1:]

        xs = []
        while True:
            x,s = Program._parse(s)
            xs.append(x)
            while len(s) > 0 and s[0].isspace(): s = s[1:]
            if s == "": raise ParseFailure(s)
            if s[0] == ")":
                s = s[1:]
                break
        e = xs[0]
        for x in xs[1:]: e = Application(e,x)
        return e,s

            

class Index(Program):
    '''
    deBruijn index: https://en.wikipedia.org/wiki/De_Bruijn_index
    These indices encode variables.
    '''
    def __init__(self,i):
        self.i = i
    def show(self,isFunction): return "$%d"%self.i
    def __eq__(self,o): return isinstance(o,Index) and o.i == self.i
    def __hash__(self): return self.i
    def visit(self, visitor, *arguments, **keywords): return visitor.index(self, *arguments, **keywords)
    def evaluate(self,environment):
        return environment[self.i]
    def inferType(self,context,environment,freeVariables):
        if self.bound(len(environment)):
            return (context, environment[self.i].apply(context))
        else:
            i = self.i - len(environment)
            if i in freeVariables:
                return (context, freeVariables[i].apply(context))
            context, variable = context.makeVariable()
            freeVariables[i] = variable
            return (context, variable)

    def shift(self,offset, depth = 0):
        # bound variable
        if self.bound(depth): return self
        else: # free variable
            i = self.i + offset
            if i < 0: raise ShiftFailure()
            return Index(i)
    def substitute(self, old, new):
        if old == self: return new
        else: return self

    def walk(self,surroundingAbstractions = 0): yield surroundingAbstractions,self
    def walkUncurried(self,d = 0): yield d,self

    def size(self): return 1

    def free(self, surroundingAbstractions):
        '''Is this index a free variable, given that it has surroundingAbstractions lambda's around it?'''
        return self.i >= surroundingAbstractions
    def bound(self, surroundingAbstractions):
        '''Is this index a bound variable, given that it has surroundingAbstractions lambda's around it?'''
        return self.i < surroundingAbstractions

    @property
    def isIndex(self): return True

    @staticmethod
    def _parse(s):
        while len(s) > 0 and s[0].isspace(): s = s[1:]
        if s == "" or s[0] != '$': raise ParseFailure(s)
        s = s[1:]
        n = ""
        while s != "" and s[0].isdigit():
            n += s[0]
            s = s[1:]
        if n == "": raise ParseFailure(s)
        return Index(int(n)),s
        


class Abstraction(Program):
    '''Lambda abstraction. Creates a new function.'''
    def __init__(self,body):
        self.body = body
        self.hashCode = None
    @property
    def isAbstraction(self): return True
    def __eq__(self,o): return isinstance(o,Abstraction) and o.body == self.body
    def __hash__(self):
        if self.hashCode == None: self.hashCode = hash((hash(self.body),))
        return self.hashCode
    def visit(self, visitor, *arguments, **keywords): return visitor.abstraction(self, *arguments, **keywords)
    def show(self,isFunction):
        return "(lambda %s)"%(self.body.show(False))
    def evaluate(self,environment):
        return lambda x: self.body.evaluate([x] + environment)
    def inferType(self,context,environment,freeVariables):
        (context,argumentType) = context.makeVariable()
        (context,returnType) = self.body.inferType(context,[argumentType] + environment,freeVariables)
        return (context, arrow(argumentType,returnType).apply(context))
    
    def shift(self,offset, depth = 0):
        return Abstraction(self.body.shift(offset, depth + 1))
    def substitute(self, old, new):
        if self == old: return new
        old = old.shift(1)
        new = new.shift(1)
        return Abstraction(self.body.substitute(old, new))
    
    def walk(self,surroundingAbstractions = 0):
        yield surroundingAbstractions,self
        for child in self.body.walk(surroundingAbstractions + 1): yield child
    def walkUncurried(self,d = 0):
        yield d,self
        for k in self.body.walkUncurried(d+1): yield k

    def size(self): return self.body.size()

    @staticmethod
    def _parse(s):
        if s.startswith('(\\'):
            s = s[2:]
        elif s.startswith('(lambda'):
            s = s[len('(lambda'):]
        elif s.startswith(u'(\u03bb'):
            s = s[len(u'(\u03bb'):]
        else: raise ParseFailure(s)
        while len(s) > 0 and s[0].isspace(): s = s[1:]

        b,s = Program._parse(s)
        while len(s) > 0 and s[0].isspace(): s = s[1:]
        if s == "" or s[0] != ')': raise ParseFailure(s)
        s = s[1:]
        return Abstraction(b),s


        

class Primitive(Program):
    GLOBALS = {}
    def __init__(self, name, ty, value):
        self.tp = ty
        self.name = name
        self.value = value
        if name not in Primitive.GLOBALS: Primitive.GLOBALS[name] = self
    @property
    def isPrimitive(self): return True
    def __eq__(self,o): return isinstance(o,Primitive) and o.name == self.name
    def __hash__(self): return hash(self.name)
    def visit(self, visitor, *arguments, **keywords): return visitor.primitive(self, *arguments, **keywords)
    def show(self,isFunction): return self.name
    def evaluate(self,environment): return self.value
    def inferType(self,context,environment,freeVariables):
        return self.tp.instantiate(context)
    def shift(self,offset, depth = 0): return self
    def substitute(self, old, new):
        if self == old: return new
        else: return self

    def walk(self,surroundingAbstractions = 0): yield surroundingAbstractions,self
    def walkUncurried(self,d = 0): yield d,self

    def size(self): return 1

    @staticmethod
    def _parse(s):
        while len(s) > 0 and s[0].isspace(): s = s[1:]
        name = ""
        while s != "" and not s[0].isspace() and s[0] not in '()':
            name += s[0]
            s = s[1:]
        if name in Primitive.GLOBALS:
            return Primitive.GLOBALS[name],s
        raise ParseFailure(s)



class Invented(Program):
    '''New invented primitives'''
    def __init__(self, body):
        self.body = body
        self.tp = self.body.infer()
        self.hashCode = None
    @property
    def isInvented(self): return True
    def show(self,isFunction): return "#%s"%(self.body.show(False))
    def visit(self, visitor, *arguments, **keywords): return visitor.invented(self, *arguments, **keywords)
    def __eq__(self,o): return isinstance(o,Invented) and o.body == self.body
    def __hash__(self):
        if self.hashCode == None: self.hashCode = hash((0,hash(self.body)))
        return self.hashCode
    def evaluate(self,e): return self.body.evaluate([])
    def inferType(self,context,environment,freeVariables):
        return self.tp.instantiate(context)
    def shift(self,offset, depth = 0): return self
    def substitute(self, old, new):
        if self == old: return new
        else: return self

    def walk(self,surroundingAbstractions = 0): yield surroundingAbstractions,self
    def walkUncurried(self,d = 0): yield d,self

    def size(self): return 1

    @staticmethod
    def _parse(s):
        while len(s) > 0 and s[0].isspace(): s = s[1:]
        if not s.startswith('#'): raise ParseFailure(s)
        s = s[1:]
        b,s = Program._parse(s)
        return Invented(b),s
    

class FragmentVariable(Program):
    def __init__(self): pass
    def show(self,isFunction): return "??"
    def __eq__(self,o): return isinstance(o,FragmentVariable)
    def __hash__(self): return 42
    def visit(self, visitor, *arguments, **keywords):
        return visitor.fragmentVariable(self, *arguments, **keywords)
    def evaluate(self, e):
        raise Exception('Attempt to evaluate fragment variable')
    def inferType(self,context, environment, freeVariables):
        return context.makeVariable()
    def shift(self,offset,depth = 0):
        raise Exception('Attempt to shift fragment variable')
    def substitute(self, old, new):
        if self == old: return new
        else: return self
    def match(self, context, expression, holes, variableBindings, environment = []):
        surroundingAbstractions = len(environment)
        try:
            context, variable = context.makeVariable()
            holes.append((variable, expression.shift(-surroundingAbstractions)))
            return context, variable
        except ShiftFailure: raise MatchFailure()

    def walk(self, surroundingAbstractions = 0): yield surroundingAbstractions,self
    def walkUncurried(self,d = 0): yield d,self

    def size(self): return 1

    @staticmethod
    def _parse(s):
        while len(s) > 0 and s[0].isspace(): s = s[1:]
        if s.startswith('??'): return FragmentVariable.single, s[2:]
        if s.startswith('?'): return FragmentVariable.single, s[1:]
        raise ParseFailure(s)

FragmentVariable.single = FragmentVariable()

class ShareVisitor(object):
    def __init__(self):
        self.primitiveTable = {}
        self.inventedTable = {}
        self.indexTable = {}
        self.applicationTable = {}
        self.abstractionTable = {}
        
    def invented(self,e):
        body = e.body.visit(self)
        i = id(body)
        if i in self.inventedTable: return self.inventedTable[i]
        new = Invented(body)
        self.inventedTable[i] = new
        return new
    def primitive(self,e):
        if e.name in self.primitiveTable: return self.primitiveTable[e.name]
        self.primitiveTable[e.name] = e
        return e
    def index(self,e):
        if e.i in self.indexTable: return self.indexTable[e.i]
        self.indexTable[e.i] = e
        return e
    def application(self,e):
        f = e.f.visit(self)
        x = e.x.visit(self)
        fi = id(f)
        xi = id(x)
        i = (fi,xi)
        if i in self.applicationTable: return self.applicationTable[i]
        new = Application(f,x)
        self.applicationTable[i] = new
        return new        
    def abstraction(self,e):
        body = e.body.visit(self)
        i = id(body)
        if i in self.abstractionTable: return self.abstractionTable[i]
        new = Abstraction(body)
        self.abstractionTable[i] = new
        return new
    def execute(self,e):
        return e.visit(self)

class Mutator:
    """Perform local mutations to an expr"""
    def __init__(self, grammar, fn):
        """Fn yields expressions from a type and loss."""
        self.fn = fn
        self.grammar = grammar
        self.history = []
    def enclose(self, expr):
        for h in self.history[::-1]:
            expr = h(expr)
        return expr
    def invented(self,e,tp):
        for expr in self.fn(tp, -self.grammar.expression2likelihood[e]):
            yield self.enclose(expr)
    def primitive(self,e,tp):
        for expr in self.fn(tp, -self.grammar.expression2likelihood[e]):
            yield self.enclose(expr)
    def index(self,e,tp):
        for expr in self.fn(tp, -self.grammar.logVariable):
            yield self.enclose(expr)
    def application(self,e,tp):
        self.history.append(lambda expr: Application(expr, e.x))
        f_tp = arrow(e.x.infer(), tp)
        for inner in e.f.visit(self,f_tp):
            yield inner
        self.history[-1] = lambda expr: Application(e.f, expr)
        x_tp = inferArg(tp, e.f.infer())
        for inner in e.x.visit(self,x_tp):
            yield inner
        self.history.pop()
        for expr in self.fn(tp, -self.logLikelihood(tp, e)):
            yield self.enclose(expr)
    def abstraction(self,e,tp):
        self.history.append(lambda expr: Abstraction(expr))
        for inner in e.body.visit(self,tp.arguments[1]):
            yield inner
        self.history.pop()
        # we don't try turning the abstraction into something else, because
        # that other thing will just be an abstraction
    def execute(self,e,tp):
        for expr in e.visit(self, tp):
            yield expr
    def logLikelihood(self, tp, e):
        summary = None
        try:
            summary = self.grammar.closedLikelihoodSummary(tp, e, silent=True)
        except AssertionError:
            pass
        if summary is not None:
            return summary.logLikelihood(self.grammar)
        else:
            depth, tmpTp = 0, tp
            while tmpTp.isArrow() and not isinstance(e, Abstraction):
                depth, tmpTp = depth + 1, tmpTp.arguments[1]
            old = e
            for _ in range(depth):
                e = Abstraction(Application(e, Index(0)))
            if e == old:
                return NEGATIVEINFINITY
            else:
                return self.logLikelihood(tp, e)

class RegisterPrimitives(object):
    def invented(self,e): e.body.visit(self)        
    def primitive(self,e):
        if e.name not in Primitive.GLOBALS:
            Primitive(e.name, e.tp, e.value)
    def index(self,e): pass
    def application(self,e):
        e.f.visit(self)
        e.x.visit(self)   
    def abstraction(self,e): e.body.visit(self)        
    @staticmethod
    def register(e): e.visit(RegisterPrimitives())
        
        
class PrettyVisitor(object):
    def __init__(self):
        self.numberOfVariables = 0
        self.freeVariables = {}

        self.variableNames = ["x","y","z","u","v","w"]
        self.variableNames += [chr(ord('a') + j)
                               for j in range(20) ]
        self.toplevel = True
    def makeVariable(self):
        v = self.variableNames[self.numberOfVariables]
        self.numberOfVariables += 1
        return v
    def invented(self,e,environment,isFunction,isAbstraction):
        s = e.body.visit(self,[],isFunction,isAbstraction)
        return s
    def primitive(self,e,environment,isVariable,isAbstraction): return e.name
    def index(self,e,environment,isVariable,isAbstraction):
        if e.i < len(environment):
            return environment[e.i]
        else:
            i = e.i - len(environment)
            if i in self.freeVariables:
                return self.freeVariables[i]
            else:
                v = self.makeVariable()
                self.freeVariables[i] = v
                return v
            
    def application(self,e,environment,isFunction,isAbstraction):
        self.toplevel = False
        s = u"%s %s"%(e.f.visit(self,environment,True,False),
                      e.x.visit(self,environment,False,False))
        if isFunction: return s
        else: return u"(" + s + u")"
    def abstraction(self,e,environment,isFunction,isAbstraction):
        toplevel = self.toplevel
        self.toplevel = False
        # Invent a new variable
        v = self.makeVariable()
        
        body = e.body.visit(self,
                            [v]+environment,
                            False,
                            True)
        if not e.body.isAbstraction:
            body = u"." + body
        body = v + body
        if not isAbstraction:
            body = u"λ" + body
        if not toplevel:
            body = u"(%s)"%body
        return body

def prettyProgram(e):
    return e.visit(PrettyVisitor(),[],True,False)
