from functools import reduce

try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from bin.rational import RandomParameterization
from dreamcoder.domains.arithmetic.arithmeticPrimitives import (
    f0, f1, fpi, real_power, real_subtraction, real_addition,
    real_division, real_multiplication)
from dreamcoder.domains.list.listPrimitives import bootstrapTarget
from dreamcoder.dreamcoder import explorationCompression, commandlineArguments
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
from dreamcoder.recognition import RecurrentFeatureExtractor, DummyFeatureExtractor
from dreamcoder.task import DifferentiableTask, squaredErrorLoss
from dreamcoder.type import baseType, tlist, arrow
from dreamcoder.utilities import eprint, numberOfCPUs

tvector = baseType("vector")
treal = baseType("real")
tpositive = baseType("positive")


def makeTrainingData(request, law,
                     # Number of examples
                     N=10,
                     # Vector dimensionality
                     D=2,
                     # Maximum absolute value of a random number
                     S=20.):
    from random import random, randint

    def sampleArgument(a, listLength):
        if a.name == "real":
            return random() * S * 2 - S
        elif a.name == "positive":
            return random() * S
        elif a.name == "vector":
            return [random() * S * 2 - S for _ in range(D)]
        elif a.name == "list":
            return [sampleArgument(a.arguments[0], listLength)
                    for _ in range(listLength)]
        else:
            assert False, "unknown argument tp %s" % a

    arguments = request.functionArguments()
    e = []
    for _ in range(N):
        # Length of any requested lists
        l = randint(1, 4)

        xs = tuple(sampleArgument(a, l) for a in arguments)
        y = law(*xs)
        e.append((xs, y))

    return e


def makeTask(name, request, law,
             # Number of examples
             N=20,
             # Vector dimensionality
             D=3,
             # Maximum absolute value of a random number
             S=20.):
    print(name)
    e = makeTrainingData(request, law,
                         N=N, D=D, S=S)
    print(e)
    print()

    def genericType(t):
        if t.name == "real":
            return treal
        elif t.name == "positive":
            return treal
        elif t.name == "vector":
            return tlist(treal)
        elif t.name == "list":
            return tlist(genericType(t.arguments[0]))
        elif t.isArrow():
            return arrow(genericType(t.arguments[0]),
                         genericType(t.arguments[1]))
        else:
            assert False, "could not make type generic: %s" % t

    return DifferentiableTask(name, genericType(request), e,
                              BIC=10.,
                              likelihoodThreshold=-0.001,
                              restarts=2,
                              steps=25,
                              maxParameters=1,
                              loss=squaredErrorLoss)


def norm(v):
    return sum(x * x for x in v)**0.5


def unit(v):
    return scaleVector(1. / norm(v), v)


def scaleVector(a, v):
    return [a * x for x in v]


def innerProduct(a, b):
    return sum(x * y for x, y in zip(a, b))


def crossProduct(a, b):
    (a1, a2, a3) = tuple(a)
    (b1, b2, b3) = tuple(b)
    return [a2 * b3 - a3 * b2,
            a3 * b1 - a1 * b3,
            a1 * b2 - a2 * b1]


def vectorAddition(u, v):
    return [a + b for a, b in zip(u, v)]

def vectorSubtraction(u, v):
    return [a - b for a, b in zip(u, v) ]


class LearnedFeatureExtractor(RecurrentFeatureExtractor):
    def tokenize(self, examples):
        # Should convert both the inputs and the outputs to lists
        def t(z):
            if isinstance(z, list):
                return ["STARTLIST"] + \
                    [y for x in z for y in t(x)] + ["ENDLIST"]
            assert isinstance(z, (float, int))
            return ["REAL"]
        return [(tuple(map(t, xs)), t(y))
                for xs, y in examples]

    def __init__(self, tasks, examples, testingTasks=[], cuda=False):
        lexicon = {c
                   for t in tasks + testingTasks
                   for xs, y in self.tokenize(t.examples)
                   for c in reduce(lambda u, v: u + v, list(xs) + [y])}

        super(LearnedFeatureExtractor, self).__init__(lexicon=list(lexicon),
                                                      cuda=cuda,
                                                      H=64,
                                                      tasks=tasks,
                                                      bidirectional=True)

    def featuresOfProgram(self, p, tp):
        p = program.visit(RandomParameterization.single)
        return super(LearnedFeatureExtractor, self).featuresOfProgram(p, tp)


if __name__ == "__main__":
    pi = 3.14  # I think this is close enough to pi
    # Data taken from:
    # https://secure-media.collegeboard.org/digitalServices/pdf/ap/ap-physics-1-equations-table.pdf
    # https://secure-media.collegeboard.org/digitalServices/pdf/ap/physics-c-tables-and-equations-list.pdf
    # http://mcat.prep101.com/wp-content/uploads/ES_MCATPhysics.pdf
    # some linear algebra taken from "parallel distributed processing"
    tasks = [
        # parallel distributed processing
        makeTask("vector addition (2)",
                 arrow(tvector, tvector, tvector),
                 vectorAddition),
        makeTask("vector addition (many)",
                 arrow(tlist(tvector), tvector),
                 lambda vs: reduce(vectorAddition, vs)),
        makeTask("vector norm",
                 arrow(tvector, treal),
                 lambda v: innerProduct(v, v) ** 0.5),
        # mcat
        makeTask("freefall velocity = (2gh)**.5",
                 arrow(tpositive, treal),
                 lambda h: (2 * 9.8 * h) ** 0.5),
        makeTask("v^2 = v_0^2 + 2a(x-x0)",
                 arrow(treal, treal, treal, treal, treal),
                 lambda v0, a, x, x0: v0 ** 2 + 2 * a * (x - x0)),
        makeTask("v = (vx**2 + vy**2)**0.5",
                 arrow(treal, treal, treal),
                 lambda vx, vy: (vx ** 2 + vy ** 2) ** 0.5),
        makeTask("a_r = v**2/R",
                 arrow(treal, tpositive, treal),
                 lambda v, r: v * v / r),
        makeTask("e = mc^2",
                 arrow(tpositive, tpositive, treal),
                 lambda m, c: m * c * c),
        makeTask("COM (general scalar)",
                 arrow(tvector, tvector, treal),
                 lambda ms, xs: sum(m * x for m, x in zip(ms, xs)) / sum(ms)),
        makeTask("COM (2 vectors)",
                 arrow(tvector, tvector, tpositive, tpositive, tvector),
                 lambda x1, x2, m1, m2: scaleVector(1. / (m1 + m2),
                                                    vectorAddition(scaleVector(m1, x1), scaleVector(m2, x2)))),
        makeTask("density = mass/volume",
                 arrow(treal, treal, treal),
                 lambda m, v: m / v),
        makeTask("pressure = force/area",
                 arrow(treal, treal, treal),
                 lambda m, v: m / v),
        makeTask("P = I^2R",
                 arrow(treal, treal, treal),
                 lambda i, r: i * i * r),
        makeTask("P = V^2/R",
                 arrow(treal, treal, treal),
                 lambda v, r: v * v / r),
        makeTask("V_{rms} = V/sqrt2",
                 arrow(treal, treal),
                 lambda v: v / (2.0 ** 0.5)),
        makeTask("U = 1/2CV^2",
                 arrow(treal, treal, treal),
                 lambda c, v: 0.5 * c * v * v),
        makeTask("U = 1/2QV",
                 arrow(treal, treal, treal),
                 lambda c, v: 0.5 * c * v),
        makeTask("U = 1/2Q^2/C",
                 arrow(treal, tpositive, treal),
                 lambda q, c: 0.5 * q * q / c),
        makeTask("P = 1/f",
                 arrow(tpositive, tpositive),
                 lambda f: 1. / f),
        makeTask("c = 1/2*r",
                 arrow(treal, treal),
                 lambda r: r / 2.),

        # AP physics
        makeTask("Fnet = sum(F)",
                 arrow(tlist(tvector), tvector),
                 lambda vs: reduce(vectorAddition, vs)),
        makeTask("a = sum(F)/m",
                 arrow(tpositive, tlist(tvector), tvector),
                 lambda m, vs: scaleVector(1. / m, reduce(vectorAddition, vs))),
        makeTask("work = F.d",
                 arrow(tvector, tvector, treal),
                 lambda f, d: innerProduct(f, d),
                 S=20.),
        makeTask("P = F.v",
                 arrow(tvector, tvector, treal),
                 lambda f, d: innerProduct(f, d),
                 S=20.),
        makeTask("F = qvxB (3d)",
                 arrow(treal, tvector, tvector, tvector),
                 lambda q, v, b: scaleVector(q, crossProduct(v, b))),
        makeTask("F = qvxB (2d)",
                 arrow(treal, treal, treal, treal, treal, treal),
                 lambda q, a1, a2, b1, b2: q * (a1 * b2 - a2 * b1)),
        makeTask("tau = rxF (3d)",
                 arrow(tvector, tvector, tvector),
                 crossProduct),
        makeTask("tau = rxF (2d)",
                 arrow(treal, treal, treal, treal, treal),
                 lambda a1, a2, b1, b2: a1 * b2 - a2 * b1),
        makeTask("v(t)",
                 arrow(treal, treal, treal, treal),
                 lambda v0, a, t: v0 + a * t),
        makeTask("x(t)",
                 arrow(treal, treal, treal, treal, treal),
                 lambda x0, v0, a, t: x0 + v0 * t + 0.5 * a * t * t),
        makeTask("p=mv",
                 arrow(tpositive, tvector, tvector),
                 lambda m, v: [m * _v for _v in v]),
        makeTask("dp=Fdt",
                 arrow(treal, tvector, tvector),
                 lambda m, v: [m * _v for _v in v]),
        makeTask("K=1/2mv^2",
                 arrow(tpositive, tvector, tpositive),
                 lambda m, v: 0.5 * m * norm(v) ** 2),
        makeTask("K=1/2Iw^2",
                 arrow(tpositive, tpositive, tpositive),
                 lambda m, v: 0.5 * m * v ** 2),
        makeTask("E=pJ",
                 arrow(treal, tvector, tvector),
                 lambda p, j: [p * _j for _j in j]),
        makeTask("Fs=kx",
                 arrow(treal, tvector, tvector),
                 lambda k, x: [k * _x for _x in x]),
        makeTask("P=dE/dt",
                 arrow(treal, treal, treal),
                 lambda de, dt: de / dt),
        makeTask("theta(t)",
                 arrow(treal, treal, treal, treal, treal),
                 lambda x0, v0, a, t: x0 + v0 * t + 0.5 * a * t * t),
        makeTask("omega(t)",
                 arrow(treal, treal, treal, treal),
                 lambda v0, a, t: v0 + a * t),
        makeTask("T=2pi/w",
                 arrow(tpositive, tpositive),
                 lambda w: 2 * pi / w),
        makeTask("Ts=2pi(m/k)^1/2",
                 arrow(tpositive, tpositive, tpositive),
                 lambda m, k: 2 * pi * (m / k) ** 0.5),
        makeTask("Tp=2pi(l/g)^1/2",
                 arrow(tpositive, tpositive, tpositive),
                 lambda m, k: 2 * pi * (m / k) ** 0.5),
        # makeTask("Newtonian gravitation (2 vectors)",
        #          arrow(tpositive, tpositive, tvector, tvector, tvector),
        #          lambda m1, m2, r1, r2: scaleVector(m1 * m2 / (norm(vectorSubtraction(r1, r2)) ** 2),
        #                                             unit(vectorSubtraction(r1, r2)))),
        makeTask("Coulomb's law (2 vectors)",
                 arrow(tpositive, tpositive, tvector, tvector, tvector),
                 lambda m1, m2, r1, r2: scaleVector(m1 * m2 / (norm(vectorSubtraction(r1, r2)) ** 2),
                                                    unit(vectorSubtraction(r1, r2)))),
        makeTask("Newtonian gravitation (vector)",
                 arrow(tpositive, tpositive, tvector, tvector),
                 lambda m1, m2, r: scaleVector(m1 * m2 / (norm(r) ** 2), unit(r))),
        makeTask("Coulomb's law (vector)",
                 arrow(tpositive, tpositive, tvector, tvector),
                 lambda m1, m2, r: scaleVector(m1 * m2 / (norm(r) ** 2), unit(r))),
        makeTask("Newtonian gravitation (scalar)",
                 arrow(tpositive, tpositive, tvector, treal),
                 lambda m1, m2, r: m1 * m2 / (norm(r) ** 2)),
        makeTask("Coulomb's law (scalar)",
                 arrow(tpositive, tpositive, tvector, treal),
                 lambda m1, m2, r: m1 * m2 / (norm(r) ** 2)),
        makeTask("Hook's law",
                 arrow(treal, tpositive, tpositive),
                 lambda k, x: -k * x * x,
                 N=20,
                 S=20),
        makeTask("Hook's law (2 vectors)",
                 arrow(treal, tvector, tvector, tpositive),
                 lambda k, u, v: k * norm(vectorSubtraction(u,v)),
                 N=20,
                 S=20),
        makeTask("Ohm's law",
                 arrow(tpositive, tpositive, tpositive),
                 lambda r, i: r * i,
                 N=20,
                 S=20),
        makeTask("power/current/voltage relation",
                 arrow(tpositive, tpositive, tpositive),
                 lambda i, v: v * i,
                 N=20,
                 S=20),
        makeTask("gravitational potential energy",
                 arrow(tpositive, treal, treal),
                 lambda m, h: 9.8 * m * h,
                 N=20,
                 S=20),
        makeTask("time/frequency relation",
                 arrow(tpositive, tpositive),
                 lambda t: 1. / t,
                 N=20,
                 S=2.),
        makeTask("Plank relation",
                 arrow(tpositive, tpositive),
                 lambda p: 1. / p,
                 N=20,
                 S=2.),
        makeTask("capacitance from charge and voltage",
                 arrow(tpositive, tpositive, tpositive),
                 lambda v, q: v / q,
                 N=20,
                 S=20),
        makeTask("series resistors",
                 arrow(tlist(tpositive), tpositive),
                 lambda cs: sum(cs),
                 N=20,
                 S=20),
        # makeTask("parallel resistors",
        #          arrow(tlist(tpositive), tpositive),
        #          lambda cs: sum(c**(-1) for c in cs)**(-1),
        #          N=20,
        #          S=20),
        makeTask("parallel capacitors",
                 arrow(tlist(tpositive), tpositive),
                 lambda cs: sum(cs),
                 N=20,
                 S=20),
        makeTask("series capacitors",
                 arrow(tlist(tpositive), tpositive),
                 lambda cs: sum(c ** (-1) for c in cs) ** (-1),
                 N=20,
                 S=20),

        makeTask("A = pir^2",
                 arrow(tpositive, tpositive),
                 lambda r: pi * r * r),
        makeTask("c^2 = a^2 + b^2",
                 arrow(tpositive, tpositive, tpositive),
                 lambda a, b: a * a + b * b)
    ]
    bootstrapTarget()
    equationPrimitives = [
#        real,
        f0,
        f1,
        fpi,
        real_power,
        real_subtraction,
        real_addition,
        real_division,
        real_multiplication] + [
            Program.parse(n)
            for n in ["map","fold",
                      "empty","cons","car","cdr",
                      "zip"]]
    baseGrammar = Grammar.uniform(equationPrimitives)

    eprint("Got %d equation discovery tasks..." % len(tasks))

    explorationCompression(baseGrammar, tasks,
                           outputPrefix="experimentOutputs/scientificLaws",
                           evaluationTimeout=0.1,
                           testingTasks=[],
                           **commandlineArguments(
                               compressor="ocaml",
                               featureExtractor=DummyFeatureExtractor,
                               iterations=10,
                               CPUs=numberOfCPUs(),
                               structurePenalty=0.5,
                               helmholtzRatio=0.5,
                               a=3,
                               maximumFrontier=10000,
                               topK=2,
                               pseudoCounts=10.0))
