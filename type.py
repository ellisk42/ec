


class UnificationFailure(Exception): pass
class Occurs(UnificationFailure): pass

class Type(object):
    def __str__(self): return self.show(True)
    def __repr__(self): return str(self)

class TypeConstructor(Type):
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

    def __eq__(self,other):
        return isinstance(other,TypeConstructor) and \
            self.name == other.name and \
            all(x == y for x,y in zip(self.arguments,other.arguments))
    def __hash__(self): return hash((self.name,) + tuple(self.arguments))
    def __ne__(self,other):
        return not (self == other)

    def show(self, isReturn):
        if self.name == ARROW:
            if isReturn: return "%s %s %s"%(self.arguments[0].show(False), ARROW, self.arguments[1].show(True))
            else: return "(%s %s %s)"%(self.arguments[0].show(False), ARROW, self.arguments[1].show(True))
        elif self.arguments == []:
            return self.name
        else:
            return "%s(%s)"%(self.name, ", ".join( x.show(True) for x in self.arguments))
        
    def isArrow(self): return self.name == ARROW

    def functionArguments(self):
        if self.name == ARROW:
            xs = self.arguments[1].functionArguments()
            return [self.arguments[0]] + xs
        return []

    def returns(self):
        if self.name == ARROW:
            return self.arguments[1].returns()
        else: return self

    def apply(self,context):
        return TypeConstructor(self.name,
                               [ x.apply(context) for x in self.arguments ])

    def occurs(self,v): return any(x.occurs(v) for x in self.arguments )

    def instantiate(self, context, bindings = None):
        if bindings == None: bindings = {}
        newArguments = []
        for x in self.arguments:
            (context,x) = x.instantiate(context, bindings)
            newArguments.append(x)
        return (context, TypeConstructor(self.name, newArguments))

    def canonical(self,bindings = None):
        if bindings == None: bindings = {}
        return TypeConstructor(self.name, [ x.canonical(bindings) for x in self.arguments ])


class TypeVariable(Type):
    def __init__(self,j):
        assert isinstance(j,int)
        self.v = j
    def __eq__(self,other):
        return isinstance(other,TypeVariable) and self.v == other.v
    def __ne__(self,other): return not (self.v == other.v)
    def __hash__(self): return self.v
    def show(self,_): return "t%d"%self.v

    def returns(self): return self
    def isArrow(self): return False

    def apply(self, context):
        for v,t in context.substitution:
            if v == self.v: return t.apply(context)
        return self

    def occurs(self,v): return v == self.v

    def instantiate(self,context, bindings = None):
        if bindings == None: bindings = {}
        if self.v in bindings: return (context,bindings[self.v])
        new = TypeVariable(context.nextVariable)
        bindings[self.v] = new
        context = Context(context.nextVariable + 1, context.substitution)
        return (context, new)

    def canonical(self,bindings = None):
        if bindings == None: bindings = {}
        if self.v in bindings: return bindings[self.v]
        new = TypeVariable(len(bindings))
        bindings[self.v] = new
        return new

    
            

class Context(object):
    def __init__(self, nextVariable = 0, substitution = []):
        self.nextVariable = nextVariable
        self.substitution = substitution
    def extend(self,j,t):
        return Context(self.nextVariable, [(j,t)] + self.substitution)
    def makeVariable(self):
        return (Context(self.nextVariable + 1, self.substitution),
                TypeVariable(self.nextVariable))
    def unify(self, t1, t2):
        t1 = t1.apply(self)
        t2 = t2.apply(self)
        if t1 == t2: return self

        if isinstance(t1,TypeVariable):
            if t2.occurs(t1.v): raise Occurs()
            return self.extend(t1.v,t2)
        if isinstance(t2,TypeVariable):
            if t1.occurs(t2.v): raise Occurs()
            return self.extend(t2.v,t1)
        if t1.name != t2.name: raise UnificationFailure()
        k = self
        for x,y in zip(t2.arguments,t1.arguments):
            k = k.unify(x,y)
        return k

Context.EMPTY = Context(0,[])

    
tint = TypeConstructor("int",[])
tbool = TypeConstructor("tbool",[])
tstring = TypeConstructor("string",[])
def tlist(t): return TypeConstructor("list",[t])
t0 = TypeVariable(0)
t1 = TypeVariable(1)
t2 = TypeVariable(2)

ARROW = "->"
def arrow(*arguments):
    if len(arguments) == 1: return arguments[0]
    return TypeConstructor(ARROW,[arguments[0],arrow(*arguments[1:])])
