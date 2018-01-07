from type import *

from time import time
import math


class MatchFailure(Exception): pass
class ShiftFailure(Exception): pass


class Program(object):
    def __repr__(self): return str(self)
    def __ne__(self,o): return not (self == o)
    def infer(self): return self.inferType(Context.EMPTY,[])[1].canonical()
    def applicationParses(self): yield self,[]

class Application(Program):
    def __init__(self,f,x):
        self.f = f
        self.x = x
    def __eq__(self,other): return isinstance(other,Application) and self.f == other.f and self.x == other.x
    def __hash__(self): return hash(self.f) + 7*hash(self.x)
    def __str__(self):
        return "(%s %s)"%(self.f,self.x)
    def evaluate(self,environment):
        return self.f.evaluate(environment)(self.x.evaluate(environment))
    def inferType(self,context,environment):
        (context,ft) = self.f.inferType(context,environment)
        (context,xt) = self.x.inferType(context,environment)
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

    def match(self, expression, variableBindings, surroundingAbstractions = 0):
        '''returns (hole bindings, ). mutates variableBindings'''
        if not isinstance(expression,Application): raise MatchFailure()
        else:
            return self.f.match(expression.f, variableBindings, surroundingAbstractions) + self.x.match(expression.x, variableBindings, surroundingAbstractions)

class Index(Program):
    def __init__(self,i):
        self.i = i
    def __str__(self):
        return "$%d"%self.i
    def __eq__(self,o): return isinstance(o,Index) and o.i == self.i
    def __hash__(self): return hash(self.i)
    def evaluate(self,environment):
        return environment[self.i]
    def inferType(self,context,environment):
        return (context, environment[self.i].apply(context))

    def shift(self,offset, depth = 0):
        # bound variable
        if self.i < depth: return self
        else: # free variable
            i = self.i + offset
            if i < 0: raise ShiftFailure()
            return Index(i)

    def match(self, expression, variableBindings, surroundingAbstractions = 0):
        # This is a bound variable
        if self.i < surroundingAbstractions:
            if expression == self: return []
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
            if variableBindings[i] != value: raise MatchFailure()
        else: variableBindings[i] = value
        return []
            
        

class Abstraction(Program):
    def __init__(self,body):
        self.body = body
    def __eq__(self,o): return isinstance(o,Abstraction) and o.body == self.body
    def __hash__(self): return 1 + hash(self.body)
    def __str__(self):
        return "(lambda %s)"%self.body
    def evaluate(self,environment):
        return lambda x: self.body.evaluate([x] + environment)
    def inferType(self,context,environment):
        (context,argumentType) = context.makeVariable()
        (context,returnType) = self.body.inferType(context,[argumentType] + environment)
        return (context, arrow(argumentType,returnType).apply(context))
    
    def shift(self,offset, depth = 0):
        return Abstraction(self.body.shift(offset, depth + 1))
    def match(self, expression, variableBindings, surroundingAbstractions = 0):
        if not isinstance(expression, Abstraction): raise MatchFailure()
        return self.body.match(expression.body, variableBindings, surroundingAbstractions + 1)

class Primitive(Program):
    GLOBALS = {}
    def __init__(self, name, ty, value):
        self.tp = ty
        self.name = name
        self.value = value
        Primitive.GLOBALS[name] = self
    def __eq__(self,o): return isinstance(o,Primitive) and o.name == self.name
    def __hash__(self): return hash(self.name)
    def __str__(self):
        return self.name
    def evaluate(self,environment): return self.value
    def inferType(self,context,environment):
        return self.tp.instantiate(context)
    def shift(self,offset, depth = 0): return self
    def match(self, expression, variableBindings, surroundingAbstractions = 0):
        if self != expression: raise MatchFailure()
        return []

class Invented(Program):
    def __init__(self, body): self.body = body
    def __str__(self): return "#(%s)"%(self.body)
    def __eq__(self,o): return isinstance(o,Invented) and o.body == self.body
    def __hash__(self): return hash(self.body) - 1
    def evaluate(self,e): return self.body.evaluate([])
    def inferType(self,context,environment):
        return self.body.inferType(context,[])
    def shift(self,offset, depth = 0): return self
    def match(self, expression, variableBindings, surroundingAbstractions = 0):
        if self != expression: raise MatchFailure()
        return []
