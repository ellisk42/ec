from type import *
from utilities import *

from time import time
import math


class MatchFailure(Exception): pass
class ShiftFailure(Exception): pass


class Program(object):
    def __repr__(self): return str(self)
    def __ne__(self,o): return not (self == o)
    def __str__(self): return self.show(False)
    def infer(self): return self.inferType(Context.EMPTY,[],{})[1].canonical()
    def applicationParses(self): yield self,[]
    @property
    def closed(self):
        for surroundingAbstractions, child in self.walk():
            if isinstance(child, FragmentVariable): return False
            if isinstance(child, Index) and child.i >= surroundingAbstractions: return False
        return True

class Application(Program):
    def __init__(self,f,x):
        self.f = f
        self.x = x
        self.hashCode = None
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

    def shift(self, offset, depth = 0):
        return Application(self.f.shift(offset, depth),
                           self.x.shift(offset, depth))
    def substitute(self, old, new):
        if self == old: return new
        return Application(self.f.substitute(old, new), self.x.substitute(old, new))

    def match(self, context, expression, holes, variableBindings, environment = []):
        '''returns (context, tp of fragment). mutates variableBindings & holes'''
        if not isinstance(expression,Application): raise MatchFailure()
        
        context,ft = self.f.match(context, expression.f, holes, variableBindings, environment)
        context,xt = self.x.match(context, expression.x, holes, variableBindings, environment)
        
        context,returnType = context.makeVariable()
        try: context = context.unify(ft,arrow(xt,returnType))
        except UnificationFailure: raise MatchFailure()
        
        return (context, returnType.apply(context))

    def walk(self,surroundingAbstractions = 0):
        yield surroundingAbstractions,self
        for child in self.f.walk(surroundingAbstractions): yield child
        for child in self.x.walk(surroundingAbstractions): yield child

    def size(self): return self.f.size() + self.x.size()

            

class Index(Program):
    def __init__(self,i):
        self.i = i
    def show(self,isFunction): return "$%d"%self.i
    def __eq__(self,o): return isinstance(o,Index) and o.i == self.i
    def __hash__(self): return self.i
    def visit(self, visitor, *arguments, **keywords): return visitor.index(self, *arguments, **keywords)
    def evaluate(self,environment):
        return environment[self.i]
    def inferType(self,context,environment,freeVariables):
        if self.i < len(environment):
            return (context, environment[self.i].apply(context))
        else:
            i = self.i - len(environment)
            if i in freeVariables: return (context, freeVariables[i].apply(context))
            context, variable = context.makeVariable()
            freeVariables[i] = variable
            return (context, variable)

    def shift(self,offset, depth = 0):
        # bound variable
        if self.i < depth: return self
        else: # free variable
            i = self.i + offset
            if i < 0: raise ShiftFailure()
            return Index(i)
    def substitute(self, old, new):
        if old == self: return new
        else: return self

    def match(self, context, expression, holes, variableBindings, environment = []):
        # This is a bound variable
        surroundingAbstractions = len(environment)
        if self.i < surroundingAbstractions:
            if expression == self:
                return (context, environment[self.i].apply(context))
            else: raise MatchFailure()
        
        # This is a free variable
        i = self.i - surroundingAbstractions
        # The value is going to be lifted out of the fragment. Make
        # sure that it doesn't refer to anything bound by a lambda in
        # the fragment.
        try:
            value = expression.shift(-surroundingAbstractions)
        except ShiftFailure: raise MatchFailure()

        # Added to the bindings
        if i in variableBindings:
            (tp,binding) = variableBindings[i]
            if binding != expression: raise MatchFailure()
        else:
            context,tp = context.makeVariable()
            variableBindings[i] = (tp,expression)
        return context,tp

    def walk(self,surroundingAbstractions = 0): yield surroundingAbstractions,self

    def size(self): return 1


    
            
        

class Abstraction(Program):
    def __init__(self,body):
        self.body = body
        self.hashCode = None
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
    def match(self, context, expression, holes, variableBindings, environment = []):
        if not isinstance(expression, Abstraction): raise MatchFailure()

        context,argumentType = context.makeVariable()
        context,returnType = self.body.match(context, expression.body, holes, variableBindings, [argumentType] + environment)

        return context, arrow(argumentType,returnType)

    def walk(self,surroundingAbstractions = 0):
        yield surroundingAbstractions,self
        for child in self.body.walk(surroundingAbstractions + 1): yield child

    def size(self): return self.body.size()

        

class Primitive(Program):
    GLOBALS = {}
    def __init__(self, name, ty, value):
        self.tp = ty
        self.name = name
        self.value = value
        if name not in Primitive.GLOBALS: Primitive.GLOBALS[name] = self
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
    def match(self, context, expression, holes, variableBindings, environment = []):
        if self != expression: raise MatchFailure()
        return self.tp.instantiate(context)

    def walk(self,surroundingAbstractions = 0): yield surroundingAbstractions,self

    def size(self): return 1


class Invented(Program):
    def __init__(self, body):
        self.body = body
        self.tp = self.body.infer()
        self.hashCode = None
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
    def match(self, context, expression, holes, variableBindings, environment = []):
        if self != expression: raise MatchFailure()
        return self.tp.instantiate(context)

    def walk(self,surroundingAbstractions = 0): yield surroundingAbstractions,self

    def size(self): return 1
    

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

    def size(self): return 1

FragmentVariable.single = FragmentVariable()
