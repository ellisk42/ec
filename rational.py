from ec import explorationCompression, commandlineArguments, Program
from fragmentGrammar import *
from grammar import Grammar
from arithmeticPrimitives import real_addition, real_multiplication, real_power, real_subtraction, real, f0, f1, fpi, real_division
from listPrimitives import bootstrapTarget
from task import DifferentiableTask, squaredErrorLoss, l1loss, Task
from type import tint, arrow
from utilities import *
from program import *
from recognition import *

import random



def makeTask(name, f):
    xs = [ x/10. for x in range(-50,50)]

    maximum = 10

    inputs = []
    outputs = []
    for x in xs:
        try: y = f(x)
        except: continue
        if abs(y) < maximum:
            inputs.append(float(x))
            outputs.append(float(y))

    if len(inputs) > 25:
        t = DifferentiableTask(name, arrow(treal,treal), [((x,),y) for x,y in zip(inputs, outputs) ],
                               BIC = 1.,
                               likelihoodThreshold=-0.5,
                               maxParameters=6,
                               loss=squaredErrorLoss)
        t.f = f
        return t
    
    return None

def randomCoefficient():
    m = 10
    if random.random() > 0.5:
        return 1. + (random.random()*(m - 1))
    return -(1. + (random.random()*(m - 1)))


def randomPolynomial(order):
    coefficients = [ randomCoefficient() for _ in range(order + 1) ]
    def f(x):
        return sum( c*(x**(order-j)) for j,c in enumerate(coefficients) )
    name = ""
    for j,c in enumerate(coefficients):
        e = order - j
        if e == 0:
            monomial = ""
        elif e == 1:
            monomial = "x"
        else: monomial = "x^%d"%e

        if j == 0:
            coefficient = "%0.1f"%c
        else:
            if c < 0:
                coefficient = " - %.01f"%(abs(c))
            else:
                coefficient = " + %.01f"%c
            
        name = name + coefficient + monomial

    return name,f

def randomFactored(order):
    offsets = [ randomCoefficient() for _ in range(order) ]
    def f(x):
        p = 1.
        for o in offsets:
            p = p*(x + o)
        return p
    name = ""
    for c in offsets:
        if c > 0:
            name += "(x + %0.1f)"%c
        else:
            name += "(x - %0.1f)"%(abs(c))
    return name,f

def randomRational():
    no = random.choice([0,1,2])
    nn,n = randomPolynomial(no)
    dn,d = randomFactored(random.choice([1,2]))

    f = lambda x: n(x)/d(x)

    if no == 0: name = "%s/[%s]"%(nn,dn)
    else: name = "(%s)/[%s]"%(nn,dn)

    return name,f

def randomPower():
    e = random.choice([1,2,3])
    c = randomCoefficient()

    def f(x):
        return c*(x**(-e))
    if e == 1:
        name = "%0.1f/x"%c
    else:
        name = "%0.1f/x^%d"%(c,e)

    return name, f

    
def drawFunction(n, dx, f, resolution=32):
    import numpy as np
    
    import matplotlib
    matplotlib.use('Agg')
    
    import matplotlib.pyplot as plot
    from scipy.misc import imresize
    
    figure = plot.figure()
    plot.plot(np.arange(-dx,dx,0.05),
              [ f(x) for x in np.arange(-dx,dx,0.05) ],
              linewidth = 20)
    plot.ylim([-10,10])
    plot.axis('off')
    figure.canvas.draw()
    data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    data = data[:,:,0]
    data = 255 - data
    data = data/255.
    # print "upper and lower bounds before resizing",np.max(data),np.min(data),data.dtype
    data = imresize(data, (resolution,resolution))/255.
    # print "upper and lower bounds after resizing",np.max(data),np.min(data),data.dtype

    plot.close(figure)

    return data

def makeTasks():
    tasks = []

    tasksPerType = 100

    for o in xrange(1,5):
        ts = []
        while len(ts) < tasksPerType:
            n,f = randomPolynomial(o)
            if makeTask(n,f) is None: continue
            ts.append(makeTask(n,f))
        tasks += ts
    ts = []
    while len(ts) < tasksPerType*3:
        n,f = randomRational()
        if makeTask(n,f) is None: continue
        ts.append(makeTask(n,f))
    tasks += ts

    ts = []
    while len(ts) < tasksPerType:
        n,f = randomPower()
        if makeTask(n,f) is None: continue
        ts.append(makeTask(n,f))
    tasks += ts
    return tasks

class RandomParameterization(object):
    def primitive(self, e):
        if e.name == 'REAL':
            return Primitive(str(e), e.tp, randomCoefficient())
        return e
    def invented(self,e): return e.body.visit(self)
    def abstraction(self,e): return Abstraction(e.body.visit(self))
    def application(self,e):
        return Application(e.f.visit(self),e.x.visit(self))
    def index(self,e): return e
RandomParameterization.single = RandomParameterization()


class FeatureExtractor(ImageFeatureExtractor):
    def __init__(self, tasks):
        super(FeatureExtractor, self).__init__(tasks)
        self.tasks = tasks

    def featuresOfTask(self,t):
        return self(t.features)
    def taskOfProgram(self,p,t):
        p = p.visit(RandomParameterization.single)
        f = lambda x: p.runWithArguments([x])
        t = makeTask(str(p), f)
        if t is None:
            return None
        t.features = map(float,list(drawFunction(200, 5., t.f).ravel()))
        delattr(t,'f')
        return t


def demo():
    from PIL import Image
    
    for j,t in enumerate(makeTasks()):#xrange(100):
        name,f = t.name, t.f #randomRational()#randomPolynomial(3)
#        if makeTask(name,f) is None: continue
        
        print j,"\n",name
        a = drawFunction(200,5.,f,resolution=128)*255
        Image.fromarray(a).convert('RGB').save("/tmp/functions/%d.png"%j)
    assert False
#demo()


if __name__ == "__main__":
    primitives = [real,
                  # f1,
                  real_division, real_addition, real_multiplication]
    baseGrammar = Grammar.uniform(primitives)
    random.seed(42)
    tasks = makeTasks()

    for t in tasks:
        t.features = map(float,list(drawFunction(200, 10., t.f).ravel()))
        delattr(t,'f')
    
    eprint("Got %d tasks..."%len(tasks))

    test, train = testTrainSplit(tasks, 100)
    eprint("Training on",len(train),"tasks")
    
    explorationCompression(baseGrammar, train,
                           outputPrefix = "experimentOutputs/rational",
                           compressor="pypy",
                           evaluationTimeout = 0.1,
                           testingTasks = test,
                           **commandlineArguments(
                               featureExtractor = FeatureExtractor,
                               iterations = 10,
                               CPUs = numberOfCPUs(),
                               structurePenalty = 1.,
                               helmholtzRatio = 0.5,
                               activation="tanh",
                               maximumFrontier = 100,
                               a = 3,
                               topK = 2,
                               pseudoCounts = 30.0))
    
