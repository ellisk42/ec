class UnificationFailure(Exception):
    pass


class Occurs(UnificationFailure):
    pass


class Type(object):
    def __str__(self): return self.show(True)

    def __repr__(self): return str(self)

    @staticmethod
    def fromjson(j):
        if "index" in j: return TypeVariable(j["index"])
        if "constructor" in j: return TypeConstructor(j["constructor"],
                                                      [ Type.fromjson(a) for a in j["arguments"] ])
        assert False


class TypeConstructor(Type):
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments
        self.isPolymorphic = any(a.isPolymorphic for a in arguments)

    def makeDummyMonomorphic(self, mapping=None):
        mapping = mapping if mapping is not None else {}
        return TypeConstructor(self.name,
                               [ a.makeDummyMonomorphic(mapping) for a in self.arguments ])

    def __eq__(self, other):
        return isinstance(other, TypeConstructor) and \
            self.name == other.name and \
            all(x == y for x, y in zip(self.arguments, other.arguments))

    def __hash__(self): return hash((self.name,) + tuple(self.arguments))

    def __ne__(self, other):
        return not (self == other)

    def show(self, isReturn):
        if self.name == ARROW:
            if isReturn:
                return "%s %s %s" % (self.arguments[0].show(
                    False), ARROW, self.arguments[1].show(True))
            else:
                return "(%s %s %s)" % (self.arguments[0].show(
                    False), ARROW, self.arguments[1].show(True))
        elif self.arguments == []:
            return self.name
        else:
            return "%s(%s)" % (self.name, ", ".join(x.show(True)
                                                    for x in self.arguments))

    def json(self):
        return {"constructor": self.name,
                "arguments": [a.json() for a in self.arguments]}


    def isArrow(self): return self.name == ARROW

    def functionArguments(self):
        if self.name == ARROW:
            xs = self.arguments[1].functionArguments()
            return [self.arguments[0]] + xs
        return []

    def returns(self):
        if self.name == ARROW:
            return self.arguments[1].returns()
        else:
            return self

    def apply(self, context):
        if not self.isPolymorphic:
            return self
        return TypeConstructor(self.name,
                               [x.apply(context) for x in self.arguments])

    def applyMutable(self, context):
        if not self.isPolymorphic:
            return self
        return TypeConstructor(self.name,
                               [x.applyMutable(context) for x in self.arguments])

    def occurs(self, v):
        if not self.isPolymorphic:
            return False
        return any(x.occurs(v) for x in self.arguments)

    def negateVariables(self):
        return TypeConstructor(self.name,
                               [a.negateVariables() for a in self.arguments])

    def instantiate(self, context, bindings=None):
        if not self.isPolymorphic:
            return context, self
        if bindings is None:
            bindings = {}
        newArguments = []
        for x in self.arguments:
            (context, x) = x.instantiate(context, bindings)
            newArguments.append(x)
        return (context, TypeConstructor(self.name, newArguments))

    def instantiateMutable(self, context, bindings=None):
        if not self.isPolymorphic:
            return self
        if bindings is None:
            bindings = {}
        newArguments = []
        return TypeConstructor(self.name, [x.instantiateMutable(context, bindings)
                                           for x in self.arguments ])
        

    def canonical(self, bindings=None):
        if not self.isPolymorphic:
            return self
        if bindings is None:
            bindings = {}
        return TypeConstructor(self.name,
                               [x.canonical(bindings) for x in self.arguments])


class TypeVariable(Type):
    def __init__(self, j):
        assert isinstance(j, int)
        self.v = j
        self.isPolymorphic = True

    def makeDummyMonomorphic(self, mapping=None):
        mapping = mapping if mapping is not None else {}
        if self.v  not in mapping:
            mapping[self.v] = TypeConstructor(f"dummy_type_{len(mapping)}", [])
        return mapping[self.v]
        

    def __eq__(self, other):
        return isinstance(other, TypeVariable) and self.v == other.v

    def __ne__(self, other): return not (self.v == other.v)

    def __hash__(self): return self.v

    def show(self, _): return "t%d" % self.v

    def json(self):
        return {"index": self.v}

    def returns(self): return self

    def isArrow(self): return False

    def functionArguments(self): return []

    def apply(self, context):
        for v, t in context.substitution:
            if v == self.v:
                return t.apply(context)
        return self

    def applyMutable(self, context):
        s = context.substitution[self.v]
        if s is None: return self
        new = s.applyMutable(context)
        context.substitution[self.v] = new
        return new

    def occurs(self, v): return v == self.v

    def instantiate(self, context, bindings=None):
        if bindings is None:
            bindings = {}
        if self.v in bindings:
            return (context, bindings[self.v])
        new = TypeVariable(context.nextVariable)
        bindings[self.v] = new
        context = Context(context.nextVariable + 1, context.substitution)
        return (context, new)

    def instantiateMutable(self, context, bindings=None):
        if bindings is None: bindings = {}
        if self.v in bindings: return bindings[self.v]
        new = context.makeVariable()
        bindings[self.v] = new
        return new

    def canonical(self, bindings=None):
        if bindings is None:
            bindings = {}
        if self.v in bindings:
            return bindings[self.v]
        new = TypeVariable(len(bindings))
        bindings[self.v] = new
        return new

    def negateVariables(self):
        return TypeVariable(-1 - self.v)


class Context(object):
    def __init__(self, nextVariable=0, substitution=[]):
        self.nextVariable = nextVariable
        self.substitution = substitution

    def extend(self, j, t):
        return Context(self.nextVariable, [(j, t)] + self.substitution)

    def makeVariable(self):
        return (Context(self.nextVariable + 1, self.substitution),
                TypeVariable(self.nextVariable))

    def unify(self, t1, t2):
        t1 = t1.apply(self)
        t2 = t2.apply(self)
        if t1 == t2:
            return self
        # t1&t2 are not equal
        if not t1.isPolymorphic and not t2.isPolymorphic:
            raise UnificationFailure(t1, t2)

        if isinstance(t1, TypeVariable):
            if t2.occurs(t1.v):
                raise Occurs()
            return self.extend(t1.v, t2)
        if isinstance(t2, TypeVariable):
            if t1.occurs(t2.v):
                raise Occurs()
            return self.extend(t2.v, t1)
        if t1.name != t2.name:
            raise UnificationFailure(t1, t2)
        k = self
        for x, y in zip(t2.arguments, t1.arguments):
            k = k.unify(x, y)
        return k

    def __str__(self):
        return "Context(next = %d, {%s})" % (self.nextVariable, ", ".join(
            "t%d ||> %s" % (k, v.apply(self)) for k, v in self.substitution))

    def __repr__(self): return str(self)

class MutableContext(object):
    def __init__(self):
        self.substitution = []

    def extend(self,i,t):
        assert self.substitution[i] is None
        self.substitution[i] = t

    def makeVariable(self):
        self.substitution.append(None)
        return TypeVariable(len(self.substitution) - 1)

    def unify(self, t1, t2):
        t1 = t1.applyMutable(self)
        t2 = t2.applyMutable(self)

        if t1 == t2: return

        # t1&t2 are not equal
        if not t1.isPolymorphic and not t2.isPolymorphic:
            raise UnificationFailure(t1, t2)

        if isinstance(t1, TypeVariable):
            if t2.occurs(t1.v):
                raise Occurs()
            self.extend(t1.v, t2)
            return 
        if isinstance(t2, TypeVariable):
            if t1.occurs(t2.v):
                raise Occurs()
            self.extend(t2.v, t1)
            return 
        if t1.name != t2.name:
            raise UnificationFailure(t1, t2)
        
        for x, y in zip(t2.arguments, t1.arguments):
            self.unify(x, y)


Context.EMPTY = Context(0, [])


def canonicalTypes(ts):
    bindings = {}
    return [t.canonical(bindings) for t in ts]


def instantiateTypes(context, ts):
    bindings = {}
    newTypes = []
    for t in ts:
        context, t = t.instantiate(context, bindings)
        newTypes.append(t)
    return context, newTypes


def baseType(n): return TypeConstructor(n, [])


tint = baseType("int")
treal = baseType("real")
tbool = baseType("bool")
tboolean = tbool  # alias
tcharacter = baseType("char")


def tlist(t): return TypeConstructor("list", [t])


def tpair(a, b): return TypeConstructor("pair", [a, b])


def tmaybe(t): return TypeConstructor("maybe", [t])


tstr = tlist(tcharacter)
t0 = TypeVariable(0)
t1 = TypeVariable(1)
t2 = TypeVariable(2)

# regex types
tpregex = baseType("pregex")

ARROW = "->"


def arrow(*arguments):
    if len(arguments) == 1:
        return arguments[0]
    return TypeConstructor(ARROW, [arguments[0], arrow(*arguments[1:])])


def inferArg(tp, tcaller):
    ctx, tp = tp.instantiate(Context.EMPTY)
    ctx, tcaller = tcaller.instantiate(ctx)
    ctx, targ = ctx.makeVariable()
    ctx = ctx.unify(tcaller, arrow(targ, tp))
    return targ.apply(ctx)


def guess_type(xs):
    """
    Return a TypeConstructor corresponding to x's python type.
    Raises an exception if the type cannot be guessed.
    """
    if all(isinstance(x, bool) for x in xs):
        return tbool
    elif all(isinstance(x, int) for x in xs):
        return tint
    elif all(isinstance(x, str) for x in xs):
        return tstr
    elif all(isinstance(x, list) for x in xs):
        return tlist(guess_type([y for ys in xs for y in ys]))
    else:
        raise ValueError("cannot guess type from {}".format(xs))


def guess_arrow_type(examples):
    a = len(examples[0][0])
    input_types = []
    for n in range(a):
        input_types.append(guess_type([xs[n] for xs, _ in examples]))
    output_type = guess_type([y for _, y in examples])
    return arrow(*(input_types + [output_type]))

def canUnify(t1, t2):
    k = MutableContext()
    t1 = t1.instantiateMutable(k)
    t2 = t2.instantiateMutable(k)
    try:
        k.unify(t1, t2)
        return True
    except UnificationFailure: return False
    
