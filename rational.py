from ec import explorationCompression, commandlineArguments, Program
from grammar import Grammar
from arithmeticPrimitives import real_addition, real_multiplication, real_power, real_subtraction, real, f0, f1, fpi, real_division
from listPrimitives import bootstrapTarget
from task import DifferentiableTask, squaredErrorLoss, l1loss, Task
from type import tint, arrow
from utilities import *
from program import *
from recognition import *

import numpy as np

import random

treal = baseType("real")



def makeTask(name, f):
    xs = [ x/10. for x in range(-50,50)]

    maximum = 10

    inputs = []
    outputs = []
    for x in xs:
        try: y = f(x)
        except: continue
        if abs(y) < maximum:
            inputs.append(x)
            outputs.append(y)

    if len(inputs) > 25:
        t = DifferentiableTask(name, arrow(treal,treal), [((x,),y) for x,y in zip(inputs, outputs) ],
                               BIC = 1.,
                               likelihoodThreshold=-0.005,
                               maxParameters=6,
                               loss=squaredErrorLoss)
        t.f = f
        return t
    
    return None

def randomPolynomial(order,m=3):
    coefficients = [ random.random()*2*m-m for _ in range(order + 1) ]
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

def randomRational():
    no = random.choice([0,1,2])
    nn,n = randomPolynomial(no)
    dn,d = randomPolynomial(random.choice([1,2]))

    f = lambda x: n(x)/d(x)

    if no == 0: name = "%s/(%s)"%(nn,dn)
    else: name = "(%s)/(%s)"%(nn,dn)

    return name,f

    
def drawFunction(n, dx, f):
    import matplotlib
    matplotlib.use('Agg')
    
    import matplotlib.pyplot as plot
    from scipy.misc import imresize
    
    figure = plot.figure()
    plot.plot(np.arange(-dx,dx,0.05),
              [ f(x) for x in np.arange(-dx,dx,0.05) ],
              linewidth = 20)
    plot.axis('off')
    figure.canvas.draw()
    data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    data = data[:,:,0]
    data[data > 250] = 0
    data = data/2.0

    data = imresize(data, (64,64))

    return data

def makeTasks():
    tasks = []

    for o in xrange(1,4):
        ts = []
        while len(ts) < 100:
            n,f = randomPolynomial(o)
            if makeTask(n,f) is None: continue
            ts.append(makeTask(n,f))
        tasks += ts
    ts = []
    while len(ts) < 100:
        n,f = randomRational()
        if makeTask(n,f) is None: continue
        ts.append(makeTask(n,f))
    tasks += ts
    return tasks
            

class FeatureExtractor(ImageFeatureExtractor):
    def __init__(self, tasks):
        super(FeatureExtractor, self).__init__(tasks)
        self.tasks = tasks

    def featuresOfTask(self,t):
        return self(t.features)
    def featuresOfProgram(self,p,t):
        assert False,"feature extractor for program not implemented for rational functions yet!!!"

def demo():
    from PIL import Image
    
    for j,t in enumerate(makeTasks()):#xrange(100):
        name,f = t.name, t.f #randomRational()#randomPolynomial(3)
#        if makeTask(name,f) is None: continue
        
        print j,"\n",name
        a = drawFunction(100,10.,f)
        Image.fromarray(a).convert('RGB').save("/tmp/functions/%d.png"%j)


if __name__ == "__main__":
    primitives = [real, 
                  real_division, real_addition, real_multiplication]
    baseGrammar = Grammar.uniform(primitives)
    random.seed(42)
    tasks = makeTasks()

    for t in tasks:
        t.features = list(drawFunction(200, 10., t.f).ravel())
        delattr(t,'f')
    
    eprint("Got %d tasks..."%len(tasks))

    test, train = testTrainSplit(tasks, 0.25)
    
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
                               maximumFrontier = 100,
                               a = 3,
                               topK = 2,
                               pseudoCounts = 10.0))
