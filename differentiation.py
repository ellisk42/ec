import math

class DN(object):
    '''differentiable node: parent object of every differentiable operation'''
    def __init__(self, arguments):
        inputs = [ x.data for x in arguments ]
        
        self.gradient = None
        self.data = self.forward(*inputs)
        self.arguments = arguments
        
        # descendents: every variable that takes this variable as input
        # descendents: [(DN,float)]
        # the additional flow parameter is d Descendent / d This
        self.descendents = []

        ds = self.backward(*inputs)
        for d,x in zip(ds,arguments):
            x.descendents.append((self,d))

    def __str__(self):
        if self.arguments == []: return self.name
        return "(%s %s)"%(self.name," ".join(str(x) for x in self.arguments ))
    def __repr__(self): return str(self)

    @property
    def derivative(self): return self.differentiate()

    def differentiate(self):
        if self.gradient is None:
            self.gradient = sum( partial * descendent.differentiate() for descendent, partial in self.descendents )
        return self.gradient
    def zeroGradients(self):
        if self.gradient is None: return
        for x in self.arguments: x.zeroGradients()
        self.gradient = None
        self.descendents = []
        if self.arguments != []: self.data = None

    def recalculate(self):
        if self.data is None:
            inputs = [ a.recalculate() for a in self.arguments ]
            self.data = self.forward(*inputs)
            partials = self.backward(*inputs)
            for d,a in zip(partials,self.arguments):
                a.descendents.append((self,d))
        return self.data

    def backPropagation(self):
        self.gradient = 1.
        self.recursivelyDifferentiate()
    def recursivelyDifferentiate(self):
        self.differentiate()
        for x in self.arguments: x.recursivelyDifferentiate()

    def updateNetwork(self):
        self.zeroGradients()
        l = self.recalculate()
        self.backPropagation()
        return l

    def log(self): return Logarithm(self)
    def square(self): return self*self
    def exp(self): return Exponentiation(self)
    def __add__(self,o): return Addition(self, Placeholder.maybe(o))
    def __radd__(self,o): return Addition(self, Placeholder.maybe(o))
    def __sub__(self,o): return Subtraction(self, Placeholder.maybe(o))
    def __rsub__(self,o): return Subtraction(Placeholder.maybe(o), self)
    def __mul__(self,o): return Multiplication(self, Placeholder.maybe(o))
    def __rmul__(self,o): return Multiplication(self, Placeholder.maybe(o))
    def __neg__(self): return Negation(self)
    def __div__(self,o): return Division(self,Placeholder.maybe(o))
    def __rdiv__(self,o): return Division(Placeholder.maybe(o),self)
    
    def gradientDescent(self, parameters, _ = None, lr, steps = 10**3, update = None):
        for j in range(steps):
            l = self.updateNetwork()
            if (not (update is None)) and j%update == 0:
                print "LOSS:",l

            for p in parameters:
                p.data -= lr*p.derivative
        

class Placeholder(DN):
    def __init__(self, initialValue = None, name = None):
        self.data = (random.random() - 0.5)*0.2 if initialValue is None else initialValue
        super(Placeholder,self).__init__([])
        self.name = name

    def __str__(self):
        if self.name is None:
            return "<%s>"%(self.data)
        return self.name

    @staticmethod
    def maybe(x):
        if isinstance(x,DN): return x
        return Placeholder(x)

    def forward(self): return self.data
    def backward(self): return []

class Addition(DN):
    def __init__(self, x, y):
        super(Addition,self).__init__([x,y])
        self.name = '+'

    def forward(self, x, y): return x + y
    def backward(self, x, y): return [1.,1.]

class Subtraction(DN):
    def __init__(self, x, y):
        super(Subtraction,self).__init__([x,y])
        self.name = '-'

    def forward(self, x, y): return x - y
    def backward(self, x, y): return [1.,-1.]

class Negation(DN):
    def __init__(self, x):
        super(Negation,self).__init__([x])
        self.name = '-'

    def forward(self, x): return -x
    def backward(self, x): return [-1.]

class Multiplication(DN):
    def __init__(self, x, y):
        super(Multiplication,self).__init__([x,y])
        self.name = '*'
    def forward(self, x, y): return x*y
    def backward(self,x,y): return [y,x]

class Division(DN):
    def __init__(self, x, y):
        super(Division,self).__init__([x,y])
        self.name = '/'
    def forward(self, x, y): return x/y
    def backward(self,x,y): return [1.0/y,-x/(y*y)]

class Exponentiation(DN):
    def __init__(self,x):
        super(Exponentiation,self).__init__([x])
        self.name = 'exp'
    def forward(self,x): return math.exp(x)
    def backward(self,x): return [math.exp(x)]

class Logarithm(DN):
    def __init__(self,x):
        super(Logarithm,self).__init__([x])
        self.name = 'log'
    def forward(self,x): return math.log(x)
    def backward(self,x): return [1./x]

class LSE(DN):
    def __init__(self, xs):
        super(LSE,self).__init__(xs)
        self.name = 'LSE'
    def forward(self,*xs):
        m = max(xs)
        return m + math.log(sum(math.exp(y - m) for y in xs))
    def backward(self,*xs):
        m = max(xs)
        zm = sum( math.exp(x - m) for x in xs )
        return [ math.exp(x - m)/zm for x in xs ]

        
        
if __name__ == "__main__":
    x = Placeholder(10.,"x")
    y = Placeholder(2.,"y")
    z = x - LSE([x,y])
    z.updateNetwork()
    print "dL/dx = %f\tdL/dy = %f"%(x.derivative,y.derivative)

    x.data = 2.
    y.data = 10.
    z.updateNetwork()
    print "dL/dx = %f\tdL/dy = %f"%(x.differentiate(),y.differentiate())

    x.data = 2.
    y.data = 2.
    z.updateNetwork()
    print "z = ",z.data,z
    print "dL/dx = %f\tdL/dy = %f"%(x.differentiate(),y.differentiate())

    loss = -z
    print loss

    
    lr = 0.001
    loss.gradientDescent([x,y],steps = 10000, update = 1000)
