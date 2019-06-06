import binutil  # required to import from lib modules

from lib.ec import explorationCompression, commandlineArguments
from lib.recognition import *

primitives = [addition, multiplication, real]

MAXIMUMCOEFFICIENT = 9
NUMBEROFEXAMPLES = 10
EXAMPLERANGE = 2.
EXAMPLES = [float(-EXAMPLERANGE + j * (2 * EXAMPLERANGE / (NUMBEROFEXAMPLES -
                                                           1)))
            for j in range(NUMBEROFEXAMPLES)]
COEFFICIENTS = [float(c) for c in range(int(-(MAXIMUMCOEFFICIENT / 2)),
                                        int((MAXIMUMCOEFFICIENT - MAXIMUMCOEFFICIENT / 2))) if c != 1]


def sign(n): return ['+', '-'][int(n < 0)]


tasks = [((a, b, c, d, e),
          DifferentiableTask("%s%dx^4 %s %dx^3 %s %dx^2 %s %dx %s %d" % (" " if a >= 0 else "", a,
                                                                         sign(b), abs(b),
                                                                         sign(c), abs(c),
                                                                         sign(d), abs(d),
                                                                         sign(e), abs(e)),
                             arrow(tint, tint),
                             [((x,), a * x * x * x * x + b * x * x * x + c * x * x + d * x + e) for x in EXAMPLES],
                             loss=squaredErrorLoss,
                             features=[float(a * x * x * x * x + b * x * x * x + c * x * x + d * x + e) for x in EXAMPLES],
                             likelihoodThreshold=-0.05))
         for a in COEFFICIENTS
         for b in COEFFICIENTS
         for c in COEFFICIENTS
         for d in COEFFICIENTS
         for e in COEFFICIENTS]


class FeatureExtractor(HandCodedFeatureExtractor):
    def _featuresOfProgram(self, program, tp):
        e = program.visit(RandomParameterization.single)
        f = e.evaluate([])
        return [float(f(x)) for x in EXAMPLES]


class RandomParameterization(object):
    def primitive(self, e):
        if e.name == 'REAL':
            v = random.choice(COEFFICIENTS)
            return Primitive(str(v), e.tp, v)
        return e

    def invented(self, e): return e.body.visit(self)

    def abstraction(self, e): return Abstraction(e.body.visit(self))

    def application(self, e):
        return Application(e.f.visit(self), e.x.visit(self))

    def index(self, e): return e


RandomParameterization.single = RandomParameterization()


class DeepFeatureExtractor(MLPFeatureExtractor):
    def __init__(self, tasks):
        super(DeepFeatureExtractor, self).__init__(tasks, H=16)

    def _featuresOfProgram(self, program, tp):
        e = program.visit(RandomParameterization.single)
        f = e.evaluate([])
        return [float(f(x)) for x in EXAMPLES]


if __name__ == "__main__":
    # Split the tasks up by the order of the polynomial
    polynomials = {}
    for coefficients, task in tasks:
        o = max([len(coefficients) - j - 1
                 for j, c in enumerate(coefficients) if c != 0] + [0])
        polynomials[o] = polynomials.get(o, [])
        polynomials[o].append(task)

    # Sample a training set
    random.seed(0)
    for p in polynomials.values():
        random.shuffle(p)
    tasks = polynomials[1][:56] + \
        polynomials[2][:44] + \
        polynomials[3][:50] + \
        polynomials[4][:50]

    baseGrammar = Grammar.uniform(primitives)
    train = tasks

    test = polynomials[1][56:(56 + 56)] + \
        polynomials[2][44:(44 + 44)] + \
        polynomials[3][50:(50 + 50)] + \
        polynomials[4][50:(50 + 50)]

    if False:
        e = Program.parse("""(lambda (+ REAL
        (* $0 (+ REAL
        (* $0 (+ REAL
        (* $0 (+ REAL
        (* $0 REAL)))))))))""")
        eprint(e)
        from lib.fragmentGrammar import *
        f = FragmentGrammar.uniform(
            baseGrammar.primitives + [Program.parse("(+ REAL $0)")])

        eprint(f.logLikelihood(arrow(tint, tint), e))
        biggest = POSITIVEINFINITY
        for t in train:
            l = t.logLikelihood(e)
            eprint(t, l)
            biggest = min(biggest, l)
        eprint(biggest)
        assert False

    if False:
        with timing("best first enumeration"):
            baseGrammar.bestFirstEnumeration(arrow(tint, tint))
        with timing("depth first search"):
            print(len(list(enumeration(baseGrammar, Context.EMPTY, [], arrow(
                tint, tint), maximumDepth=99, upperBound=13, lowerBound=0))))
        assert False

    explorationCompression(baseGrammar, train,
                           outputPrefix="experimentOutputs/regression",
                           evaluationTimeout=None,
                           testingTasks=test,
                           **commandlineArguments(frontierSize=10**2,
                                                  iterations=10,
                                                  CPUs=numberOfCPUs(),
                                                  structurePenalty=1.,
                                                  helmholtzRatio=0.5,
                                                  a=1,  # arity
                                                  maximumFrontier=1000,
                                                  topK=2,
                                                  featureExtractor=DeepFeatureExtractor,
                                                  pseudoCounts=10.0))
