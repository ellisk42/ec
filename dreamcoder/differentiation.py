import math
import random
from dreamcoder.utilities import *


class InvalidLoss(Exception):
    pass


class DN(object):
    '''differentiable node: parent object of every differentiable operation'''

    def __init__(self, arguments):
        self.gradient = None
        if arguments != []:
            self.data = None
        self.arguments = arguments

        # descendents: every variable that takes this variable as input
        # descendents: [(DN,float)]
        # the additional float parameter is d Descendent / d This
        self.descendents = []

        self.recalculate()

    def __str__(self):
        if self.arguments == []:
            return self.name
        return "(%s %s)" % (self.name, " ".join(str(x)
                                                for x in self.arguments))

    def __repr__(self):
        return "DN(op = %s, data = %s, grad = %s, #descendents = %d, args = %s)" % (
            self.name, self.data, self.gradient, len(self.descendents), self.arguments)

    @property
    def derivative(self): return self.differentiate()

    def differentiate(self):
        if self.gradient is None:
            self.gradient = sum(partial * descendent.differentiate()
                                for descendent, partial in self.descendents)
        return self.gradient

    def zeroEverything(self):
        if self.gradient is None and self.descendents == [] and (
                self.data is None or self.arguments == []):
            return

        self.gradient = None
        self.descendents = []
        if self.arguments != []:
            self.data = None

        for x in self.arguments:
            x.zeroEverything()

    def lightweightRecalculate(self):
        return self.forward(*[a.lightweightRecalculate()
                              for a in self.arguments])

    def recalculate(self):
        if self.data is None:
            inputs = [a.recalculate() for a in self.arguments]
            self.data = self.forward(*inputs)
            # if invalid(self.data):
            #     eprint("I am invalid",repr(self))
            #     eprint("Here are my inputs",inputs)
            #     self.zeroEverything()
            #     eprint("Here I am after being zeroed",repr(self))
            #     raise Exception('invalid loss')
            #assert valid(self.data)
            partials = self.backward(*inputs)
            for d, a in zip(partials, self.arguments):
                # if invalid(d):
                #     eprint("I have an invalid derivative",self)
                #     eprint("Inputs",inputs)
                #     eprint("partials",partials)
                #     raise Exception('invalid derivative')
                a.descendents.append((self, d))
        return self.data

    def backPropagation(self):
        self.gradient = 1.
        self.recursivelyDifferentiate()

    def recursivelyDifferentiate(self):
        self.differentiate()
        for x in self.arguments:
            x.recursivelyDifferentiate()

    def updateNetwork(self):
        self.zeroEverything()
        l = self.recalculate()
        self.backPropagation()
        return l

    def log(self): return Logarithm(self)

    def square(self): return Square(self)

    def exp(self): return Exponentiation(self)

    def clamp(self, l, u): return Clamp(self, l, u)

    def __abs__(self): return AbsoluteValue(self)

    def __add__(self, o): return Addition(self, Placeholder.maybe(o))

    def __radd__(self, o): return Addition(self, Placeholder.maybe(o))

    def __sub__(self, o): return Subtraction(self, Placeholder.maybe(o))

    def __rsub__(self, o): return Subtraction(Placeholder.maybe(o), self)

    def __mul__(self, o): return Multiplication(self, Placeholder.maybe(o))

    def __rmul__(self, o): return Multiplication(self, Placeholder.maybe(o))

    def __neg__(self): return Negation(self)

    def __truediv__(self, o): return Division(self, Placeholder.maybe(o))

    def __rtruediv__(self, o): return Division(Placeholder.maybe(o), self)

    def numericallyVerifyGradients(self, parameters):
        calculatedGradients = [p.derivative for p in parameters]
        e = 0.00001
        for j, p in enumerate(parameters):
            p.data -= e
            y1 = self.lightweightRecalculate()
            p.data += 2 * e
            y2 = self.lightweightRecalculate()
            p.data -= e
            d = (y2 - y1) / (2 * e)
            if abs(calculatedGradients[j] - d) > 0.1:
                eprint(
                    "Bad gradient: expected %f, got %f" %
                    (d, calculatedGradients[j]))

    def gradientDescent(
            self,
            parameters,
            _=None,
            lr=0.001,
            steps=10**3,
            update=None):
        for j in range(steps):
            l = self.updateNetwork()
            if update is not None and j % update == 0:
                eprint("LOSS:", l)
                for p in parameters:
                    eprint(p.data, '\t', p.derivative)
            if invalid(l):
                raise InvalidLoss()

            for p in parameters:
                p.data -= lr * p.derivative
        return self.data

    def restartingOptimize(self, parameters, _=None, attempts=1,
                           s=1., decay=0.5, grow=0.1,
                           lr=0.1, steps=10**3, update=None):
        ls = []
        for _ in range(attempts):
            for p in parameters:
                p.data = random.random()*10 - 5
            ls.append(
                self.resilientBackPropagation(
                    parameters, lr=lr, steps=steps,
                    decay=decay, grow=grow))
        return min(ls)

    def resilientBackPropagation(
            self,
            parameters,
            _=None,
            decay=0.5,
            grow=1.2,
            lr=0.1,
            steps=10**3,
            update=None):
        previousSign = [None] * len(parameters)
        lr = [lr] * len(parameters)
        for j in range(steps):
            l = self.updateNetwork()

            if update is not None and j % update == 0:
                eprint("LOSS:", l)
                eprint("\t".join(str(p.derivative) for p in parameters))
            if invalid(l):
                raise InvalidLoss()

            newSigns = [p.derivative > 0 for p in parameters]
            for i, p in enumerate(parameters):
                if p.derivative > 0:
                    p.data -= lr[i]
                elif p.derivative < 0:
                    p.data += lr[i]
                if previousSign[i] is not None:
                    if previousSign[i] == newSigns[i]:
                        lr[i] *= grow
                    else:
                        lr[i] *= decay
            previousSign = newSigns

        return self.data


class Placeholder(DN):
    COUNTER = 0

    def __init__(self, initialValue=0., name=None):
        self.data = initialValue
        super(Placeholder, self).__init__([])
        if name is None:
            name = "p_" + str(Placeholder.COUNTER)
            Placeholder.COUNTER += 1
        self.name = name

    @staticmethod
    def named(namePrefix, initialValue=0.):
        p = Placeholder(initialValue, namePrefix + str(Placeholder.COUNTER))
        Placeholder.COUNTER += 1
        return p

    def __str__(self):
        return "Placeholder(%s = %s)" % (self.name, self.data)

    @staticmethod
    def maybe(x):
        if isinstance(x, DN):
            return x
        return Placeholder(float(x))

    def forward(self): return self.data

    def backward(self): return []


class Clamp(DN):
    def __init__(self, x, l, u):
        assert u > l
        self.l = l
        self.u = u
        super(Clamp, self).__init__([x])
        self.name = "clamp"

    def forward(self, x):
        if x > self.u:
            return self.u
        if x < self.l:
            return self.l
        return x

    def backward(self, x):
        if x > self.u or x < self.l:
            return [0.]
        else:
            return [1.]


class Addition(DN):
    def __init__(self, x, y):
        super(Addition, self).__init__([x, y])
        self.name = '+'

    def forward(self, x, y): return x + y

    def backward(self, x, y): return [1., 1.]


class Subtraction(DN):
    def __init__(self, x, y):
        super(Subtraction, self).__init__([x, y])
        self.name = '-'

    def forward(self, x, y): return x - y

    def backward(self, x, y): return [1., -1.]


class Negation(DN):
    def __init__(self, x):
        super(Negation, self).__init__([x])
        self.name = '-'

    def forward(self, x): return -x

    def backward(self, x): return [-1.]


class AbsoluteValue(DN):
    def __init__(self, x):
        super(AbsoluteValue, self).__init__([x])
        self.name = 'abs'

    def forward(self, x): return abs(x)

    def backward(self, x):
        if x > 0:
            return [1.]
        return [-1.]


class Multiplication(DN):
    def __init__(self, x, y):
        super(Multiplication, self).__init__([x, y])
        self.name = '*'

    def forward(self, x, y): return x * y

    def backward(self, x, y): return [y, x]


class Division(DN):
    def __init__(self, x, y):
        super(Division, self).__init__([x, y])
        self.name = '/'

    def forward(self, x, y): return x / y

    def backward(self, x, y): return [1.0 / y, -x / (y * y)]


class Square(DN):
    def __init__(self, x):
        super(Square, self).__init__([x])
        self.name = 'sq'

    def forward(self, x): return x * x

    def backward(self, x): return [2 * x]


class Exponentiation(DN):
    def __init__(self, x):
        super(Exponentiation, self).__init__([x])
        self.name = 'exp'

    def forward(self, x): return math.exp(x)

    def backward(self, x): return [math.exp(x)]


class Logarithm(DN):
    def __init__(self, x):
        super(Logarithm, self).__init__([x])
        self.name = 'log'

    def forward(self, x): return math.log(x)

    def backward(self, x): return [1. / x]


class LSE(DN):
    def __init__(self, xs):
        super(LSE, self).__init__(xs)
        self.name = 'LSE'

    def forward(self, *xs):
        m = max(xs)
        return m + math.log(sum(math.exp(y - m) for y in xs))

    def backward(self, *xs):
        m = max(xs)
        zm = sum(math.exp(x - m) for x in xs)
        return [math.exp(x - m) / zm for x in xs]


if __name__ == "__main__":
    x = Placeholder(10., "x")
    y = Placeholder(2., "y")
    z = x - LSE([x, y])
    z.updateNetwork()
    eprint("dL/dx = %f\tdL/dy = %f" % (x.derivative, y.derivative))

    x.data = 2.
    y.data = 10.
    z.updateNetwork()
    eprint("dL/dx = %f\tdL/dy = %f" % (x.differentiate(), y.differentiate()))

    x.data = 2.
    y.data = 2.
    z.updateNetwork()
    eprint("z = ", z.data, z)
    eprint("dL/dx = %f\tdL/dy = %f" % (x.differentiate(), y.differentiate()))

    loss = -z
    eprint(loss)

    lr = 0.001
    loss.gradientDescent([x, y], steps=10000, update=1000)
