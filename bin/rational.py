import datetime
import os
import random

try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.dreamcoder import explorationCompression, commandlineArguments
from dreamcoder.domains.arithmetic.arithmeticPrimitives import real, real_division, real_addition, real_multiplication
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive, Abstraction, Application
from dreamcoder.recognition import ImageFeatureExtractor
from dreamcoder.task import DifferentiableTask, squaredErrorLoss
from dreamcoder.type import arrow, treal
from dreamcoder.utilities import testTrainSplit, eprint, numberOfCPUs


def makeTask(name, f, actualParameters):
    xs = [x / 100. for x in range(-500, 500)]

    maximum = 10

    N = 50

    inputs = []
    outputs = []
    for x in xs:
        try:
            y = f(x)
        except BaseException:
            continue
        if abs(y) < maximum:
            inputs.append(float(x))
            outputs.append(float(y))

    if len(inputs) >= N:
        ex = list(zip(inputs, outputs))
        ex = ex[::int(len(ex) / N)][:N]
        t = DifferentiableTask(name,
                               arrow(treal, treal),
                               [((x,),y) for x, y in ex],
                               BIC=1.,
                               restarts=360, steps=50,
                               likelihoodThreshold=-0.05,
                               temperature=0.1,
                               actualParameters=actualParameters,
                               maxParameters=6,
                               loss=squaredErrorLoss)
        t.f = f
        return t

    return None


def randomCoefficient(m=5):
    t = 0.3
    f = t + (random.random() * (m - t))
    if random.random() > 0.5:
        f = -f
    f = float("%0.1f" % f)
    return f

def randomOffset():
    c = randomCoefficient(m=2.5)
    def f(x): return x + c
    name = "x + %0.1f" % c
    return name, f

def randomPolynomial(order):
    coefficients = [randomCoefficient(m=2.5) for _ in range(order + 1)]

    def f(x):
        return sum(c * (x**(order - j)) for j, c in enumerate(coefficients))
    name = ""
    for j, c in enumerate(coefficients):
        e = order - j
        if e == 0:
            monomial = ""
        elif e == 1:
            monomial = "x"
        else:
            monomial = "x^%d" % e

        if j == 0:
            coefficient = "%0.1f" % c
        else:
            if c < 0:
                coefficient = " - %.01f" % (abs(c))
            else:
                coefficient = " + %.01f" % c

        name = name + coefficient + monomial

    return name, f


def randomFactored(order):
    offsets = [randomCoefficient(m=5) for _ in range(order)]

    def f(x):
        p = 1.
        for o in offsets:
            p = p * (x + o)
        return p
    name = ""
    for c in offsets:
        if c > 0:
            name += "(x + %0.1f)" % c
        else:
            name += "(x - %0.1f)" % (abs(c))
    return name, f


def randomRational():
    no = random.choice([0, 1])
    nn, n = randomPolynomial(no)
    nf = random.choice([1, 2])
    dn, d = randomFactored(nf)

    def f(x): return n(x) / d(x)

    if no == 0:
        name = "%s/[%s]" % (nn, dn)
    else:
        name = "(%s)/[%s]" % (nn, dn)

    return name, f, no + 1 + nf


def randomPower():
    e = random.choice([1, 2, 3])
    c = randomCoefficient()

    def f(x):
        return c * (x**(-e))
    if e == 1:
        name = "%0.1f/x" % c
    else:
        name = "%0.1f/x^%d" % (c, e)

    return name, f

def prettyFunction(f, export):
    import numpy as np
    n = 200
    dx = 10.

    import matplotlib
    #matplotlib.use('Agg')

    import matplotlib.pyplot as plot
    from scipy.misc import imresize

    figure = plot.figure()
    plot.plot(np.arange(-dx, dx, 0.05),
              [0.5*f(x/2) for x in np.arange(-dx, dx, 0.05)],
              linewidth=15,
              color='c')
    plot.ylim([-dx,dx])
    plot.gca().set_xticklabels([])
    plot.gca().set_yticklabels([])
    for tic in plot.gca().xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
#    plot.xlabel([])
    #plot.yticks([])
    #plot.axis('off')
    plot.grid(color='k',linewidth=2)
    plot.savefig(export)
    print(export)
    plot.close(figure)


def drawFunction(n, dx, f, resolution=64):
    import numpy as np

    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plot
    from scipy.misc import imresize

    figure = plot.figure()
    plot.plot(np.arange(-dx, dx, 0.05),
              [f(x) for x in np.arange(-dx, dx, 0.05)],
              linewidth=20)
    plot.ylim([-10, 10])
    plot.axis('off')
    figure.canvas.draw()
    data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    data = data[:, :, 0]
    data = 255 - data
    data = data / 255.
    # print "upper and lower bounds before
    # resizing",np.max(data),np.min(data),data.dtype
    data = imresize(data, (resolution, resolution)) / 255.
    # print "upper and lower bounds after
    # resizing",np.max(data),np.min(data),data.dtype

    plot.close(figure)

    return data


def makeTasks():
    tasks = []

    tasksPerType = 35

    ts = []
    while len(ts) < tasksPerType:
        n, f = randomOffset()
        if makeTask(n, f, 1) is None:
            continue
        ts.append(makeTask(n, f, 1))
    tasks += ts

    for o in range(1, 5):
        ts = []
        while len(ts) < tasksPerType:
            n, f = randomPolynomial(o)
            if makeTask(n, f, o + 1) is None:
                continue
            ts.append(makeTask(n, f, o + 1))
        tasks += ts
    ts = []
    while len(ts) < tasksPerType * 3:
        n, f, df = randomRational()
        if makeTask(n, f, df) is None:
            continue
        ts.append(makeTask(n, f, df))
    tasks += ts

    ts = []
    while len(ts) < tasksPerType:
        n, f = randomPower()
        if makeTask(n, f, 1) is None:
            continue
        ts.append(makeTask(n, f, 1))
    tasks += ts
    return tasks


class RandomParameterization(object):
    def primitive(self, e):
        if e.name == 'REAL':
            return Primitive(str(e), e.tp, randomCoefficient())
        return e

    def invented(self, e): return e.body.visit(self)

    def abstraction(self, e): return Abstraction(e.body.visit(self))

    def application(self, e):
        return Application(e.f.visit(self), e.x.visit(self))

    def index(self, e): return e


RandomParameterization.single = RandomParameterization()


class FeatureExtractor(ImageFeatureExtractor):
    special = 'differentiable'
    
    def __init__(self, tasks, testingTasks=[], cuda=False, H=64):
        self.recomputeTasks = True
        super(FeatureExtractor, self).__init__(inputImageDimension=64,
                                               channels=1)
        self.tasks = tasks

    def featuresOfTask(self, t):
        return self(t.features)

    def taskOfProgram(self, p, t):
        p = p.visit(RandomParameterization.single)

        def f(x): return p.runWithArguments([x])
        t = makeTask(str(p), f, None)
        if t is None:
            return None
        t.features = drawFunction(200, 5., t.f)
        delattr(t, 'f')
        return t


def demo():
    from PIL import Image

    os.system("mkdir  -p /tmp/rational_demo")

    for j, t in enumerate(makeTasks()):  # range(100):
        name, f = t.name, t.f

        prettyFunction(f, f"/tmp/rational_demo/{name.replace('/','$')}.png")
        print(j, "\n", name)
        # a = drawFunction(200, 5., f, resolution=32) * 255
        # Image.fromarray(a).convert('RGB').save("/tmp/functions/%d.png" % j)
    assert False
#demo()    

def rational_options(p):
    p.add_argument("--smooth", action="store_true",
                   default=False,
                   help="smooth likelihood model")


if __name__ == "__main__":
    import time

    arguments = commandlineArguments(
        featureExtractor=FeatureExtractor,
        iterations=6,
        CPUs=numberOfCPUs(),
        structurePenalty=1.,
        recognitionTimeout=7200,
        helmholtzRatio=0.5,
        activation="tanh",
        maximumFrontier=5,
        a=3,
        topK=2,
        pseudoCounts=30.0,
        extras=rational_options)

    primitives = [real,
                  # f1,
                  real_division, real_addition, real_multiplication]
    baseGrammar = Grammar.uniform(primitives)
    random.seed(42)
    tasks = makeTasks()

    smooth = arguments.pop('smooth')

    for t in tasks:
        t.features = drawFunction(200, 10., t.f)
        delattr(t, 'f')
        if smooth:
            t.likelihoodThreshold = None

    eprint("Got %d tasks..." % len(tasks))

    test, train = testTrainSplit(tasks, 100)
    random.shuffle(test)
    test = test[:100]
    eprint("Training on", len(train), "tasks")

    if False:
        hardTasks = [t for t in train
                     if '/' in t.name and '[' in t.name]
        for clamp in [True, False]:
            for lr in [0.1, 0.05, 0.5, 1.]:
                for steps in [50, 100, 200]:
                    for attempts in [10, 50, 100, 200]:
                        for s in [0.1, 0.5, 1, 3]:
                            start = time.time()
                            losses = callCompiled(
                                debugMany, hardTasks, clamp, lr, steps, attempts, s)
                            losses = dict(zip(hardTasks, losses))
                            failures = 0
                            for t, l in sorted(
                                    losses.items(), key=lambda t_l: t_l[1]):
                                # print t,l
                                if l > -t.likelihoodThreshold:
                                    failures += 1
                            eprint("clamp,lr,steps, attempts,std",
                                   clamp, lr, steps, attempts, s)
                            eprint(
                                "%d/%d failures" %
                                (failures, len(hardTasks)))
                            eprint("dt=", time.time() - start)
                            eprint()
                            eprint()

        assert False
    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/rational/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)

    explorationCompression(baseGrammar, train,
                           outputPrefix="%s/rational"%outputDirectory,
                           evaluationTimeout=0.1,
                           testingTasks=test,
                           **arguments)
