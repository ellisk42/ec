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
from dreamcoder.program import Program, Primitive
from dreamcoder.recognition import RecurrentFeatureExtractor, DummyFeatureExtractor
from dreamcoder.task import DifferentiableTask, squaredErrorLoss
from dreamcoder.type import baseType, tlist, arrow, t0, t1
from dreamcoder.utilities import eprint, numberOfCPUs, flatten
import math

tvector = baseType("vector")
treal = baseType("real")

def makeTrainingData(request, law,
                     # Number of examples
                     N=10,
                     # Vector dimensionality, default is 3
                     D=3,
                     # Maximum absolute value of a random number
                     S=5.,
                     # is the length of the list fixed at 3?
                     triple=False):
    from random import random, randint

    def sampleArgument(a, listLength):
        if a.name == "real":
            return random() * S * 2 - S
        elif a.name == "vector":
            return [random() * S * 2 - S for _ in range(D)]
        elif a.name == "list":
            if triple is True:
                return [sampleArgument(a.arguments[0], 3)
                        for _ in range(3)]
            else:
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
             S=5.,
             triple=False):
    print(name)
    e = makeTrainingData(request, law,
                         N=N, D=D, S=S, triple=triple)
    print(e)
    print()

    def genericType(t):
        if t.name == "real":
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

# class LearnedFeatureExtractor(RecurrentFeatureExtractor):
#     def tokenize(self, examples):
#         # Should convert both the inputs and the outputs to lists
#         def t(z):
#             if isinstance(z, list):
#                 return ["STARTLIST"] + \
#                     [y for x in z for y in t(x)] + ["ENDLIST"]
#             assert isinstance(z, (float, int))
#             return ["REAL"]
#         return [(tuple(map(t, xs)), t(y))
#                 for xs, y in examples]

#     def __init__(self, tasks, examples, testingTasks=[], cuda=False):
#         lexicon = {c
#                    for t in tasks + testingTasks
#                    for xs, y in self.tokenize(t.examples)
#                    for c in reduce(lambda u, v: u + v, list(xs) + [y])}

#         super(LearnedFeatureExtractor, self).__init__(lexicon=list(lexicon),
#                                                       cuda=cuda,
#                                                       H=64,
#                                                       tasks=tasks,
#                                                       bidirectional=True)

#     def featuresOfProgram(self, p, tp):
#         p = program.visit(RandomParameterization.single)
#         return super(LearnedFeatureExtractor, self).featuresOfProgram(p, tp)

class LearnedFeatureExtractor(RecurrentFeatureExtractor):
    H = 64
    
    special = None

    def tokenize(self, examples):
        def sanitize(l): return [z if z in self.lexicon else "?"
                                 for z_ in l
                                 for z in (z_ if isinstance(z_, list) else [z_])]

        tokenized = []
        for xs, y in examples:
            if isinstance(y, list):
                y = ["LIST_START"] + y + ["LIST_END"]
            else:
                y = [y]
            y = sanitize(y)
            if len(y) > self.maximumLength:
                return None

            serializedInputs = []
            for xi, x in enumerate(xs):
                if isinstance(x, list):
                    x = ["LIST_START"] + x + ["LIST_END"]
                else:
                    x = [x]
                x = sanitize(x)
                if len(x) > self.maximumLength:
                    return None
                serializedInputs.append(x)

            tokenized.append((tuple(serializedInputs), y))

        return tokenized

    def __init__(self, tasks, testingTasks=[], cuda=False):
        self.lexicon = set(flatten((t.examples for t in tasks + testingTasks), abort=lambda x: isinstance(
            x, str))).union({"LIST_START", "LIST_END", "?"})

        # Calculate the maximum length
        self.maximumLength = float('inf') # Believe it or not this is actually important to have here
        self.maximumLength = max(len(l)
                                 for t in tasks + testingTasks
                                 for xs, y in self.tokenize(t.examples)
                                 for l in [y] + [x for x in xs])

        self.recomputeTasks = True

        super(
            LearnedFeatureExtractor,
            self).__init__(
            lexicon=list(
                self.lexicon),
            tasks=tasks,
            cuda=cuda,
            H=self.H,
            bidirectional=True)

    def featuresOfProgram(self, p, tp):
        p = program.visit(RandomParameterization.single)
        return super(LearnedFeatureExtractor, self).featuresOfProgram(p, tp)

def vectorAddition(u, v):
    return [a + b for a, b in zip(u, v)]

# Inverse problem
def vectorAdd1(v):
    n = len(v)
    u = [1] * n
    return [a + b for a, b in zip(u, v)]

def vectorAddFraction(v):
    n = len(v)
    u = [0.3] * n
    return [a + b for a, b in zip(u, v)]

def add1(x):
    return x + 1

def addFraction(x):
    return x + 0.3

def addFractionMed(x):
    return x + 0.375

def addFractionHard(x):
    return x + 0.375612

def vectorMultiplication(u, v):
    return [a * b for a, b in zip(u, v)]

def vectorSum(v):
    return sum(v)

def dotProduct(a, b):
    return sum(x * y for x, y in zip(a, b))

def scaleVector(a, v):
    return [a * x for x in v]

def sum_matrix_rows(m):
    return list(map(vectorSum, m))

# multiplying a list of lists with a list
def _matrix_mult(m, l): 
    # map dotProduct(l) with m so dot product is computed along rows of m and stored in a list
    return [dotProduct(m[0], l), dotProduct(m[1], l), dotProduct(m[2], l)]

def _matrix_mult_vectors(l, r1, r2, r3):
    return [dotProduct(r1, l), dotProduct(r2, l), dotProduct(r3, l)]

# activations
def _sigmoid(x): return 1 / (1 + math.exp(-x))
def _tanh(x): return math.tanh(x)
def _relu(x): return max(0.0, x)

prim_sigmoid = Primitive("sigmoid", arrow(treal, treal), _sigmoid)
prim_tanh = Primitive("tanh", arrow(treal, treal), _tanh)
prim_relu = Primitive("relu", arrow(treal, treal), _relu)

def applySigmoid(l):
    return list(map(_sigmoid, l))

def applyTanh(l):
    return list(map(_tanh, l))

def _forget_gate_noActivation(htm1, ctm1, xt, Uf, Wf, bf):
    # ft=sigmoid(Ufht−1+Wfxt+bf)
    #return lambda Uf: lambda Wf: lambda bf: list(map(_sigmoid, Uf @ htm1 + np.dot(Wf, xt) + bf))
    v1 = _matrix_mult(Uf, htm1)
    v2 = scaleVector(xt, Wf)
    return vectorAddition(bf, vectorAddition(v1, v2))

def _forget_gate(htm1, ctm1, xt, Uf, Wf, bf):
    return list(map(_sigmoid, _forget_gate_noActivation(htm1, ctm1, xt, Uf, Wf, bf)))

def _candidate_cell_state(htm1, ctm1, xt, Uc, Wc, bc):
    # ctilda=tanh(Ucht−1+Wcxt+bc)
    #return lambda Uc: lambda Wc: lambda bc: list(map(_tanh, Uc @ htm1 + np.dot(Wc, xt) + bc))
    v1 = _matrix_mult(Uc, htm1)
    v2 = scaleVector(xt, Wc)
    return list(map(_tanh, vectorAddition(bc, vectorAddition(v1, v2))))

def _new_cell_state(ft, ctm1, it, ctilda):
    # ct=ft∗ct−1+it∗ctilda
    return vectorAddition(vectorMultiplication(ft,ctm1), vectorMultiplication(it,ctilda))

def _new_hidden_state(ot, ct):
    # ht=ot∗tanh(ct)
    return vectorMultiplication(ot, list(map(_tanh, ct)))

def _lstm_cell_state(htm1, ctm1, xt, Uf, Wf, bf, Ui, Wi, bi, Uc, Wc, bc):
    ft = _forget_gate(htm1, ctm1, xt, Uf, Wf, bf)
    it = _forget_gate(htm1, ctm1, xt, Ui, Wi, bi)
    ctilda = _candidate_cell_state(htm1, ctm1, xt, Uc, Wc, bc)
    ct = _new_cell_state(ft, ctm1, it, ctilda)
    return ct

def _lstm_cell_state_noActivation(htm1, ctm1, xt, Uf, Wf, bf, Ui, Wi, bi, Uc, Wc, bc):
    ft = _forget_gate_noActivation(htm1, ctm1, xt, Uf, Wf, bf)
    it = _forget_gate_noActivation(htm1, ctm1, xt, Ui, Wi, bi)
    ctilda = _forget_gate_noActivation(htm1, ctm1, xt, Uc, Wc, bc)
    ct = _new_cell_state(ft, ctm1, it, ctilda)
    return ct

def _lstm_cell(htm1, ctm1, xt, Uf, Wf, bf, Ui, Wi, bi, Uc, Wc, bc, Uo, Wo, bo):
    ft = _forget_gate(htm1, ctm1, xt, Uf, Wf, bf)
    it = _forget_gate(htm1, ctm1, xt, Ui, Wi, bi)
    ctilda = _candidate_cell_state(htm1, ctm1, xt, Uc, Wc, bc)
    ot = _forget_gate(htm1, ctm1, xt, Uo, Wo, bo)
    ct = _new_cell_state(ft, ctm1, it, ctilda)
    ht = _new_hidden_state(ot, ct)
    return [ht, ct]

def _trunc8f(x): 
    return (math.trunc(1e8*x))*1.0e-8
prim_trunc8f = Primitive("trunc8f", arrow(treal, treal), _trunc8f)

def _trunc4f(x): 
    return (math.trunc(1e4*x))*1.0e-4
prim_trunc4f = Primitive("trunc4f", arrow(treal, treal), _trunc4f)

if __name__ == "__main__":

    tasks = [
        # parallel distributed processing
        makeTask("vector addition (2)",
                 arrow(tvector, tvector, tvector),
                 vectorAddition),
        makeTask("vector addition with (1, 1, 1)",
                 arrow(tvector, tvector),
                 vectorAdd1),
        makeTask("vector addition with (0.3, 0.3, 0.3)",
                 arrow(tvector, tvector),
                 vectorAddFraction),
        makeTask("addition with 1",
                 arrow(treal, treal),
                 add1),
        makeTask("addition with 0.3",
                 arrow(treal, treal),
                 addFraction),
        makeTask("addition with 0.375",
                 arrow(treal, treal),
                 addFractionMed),
        makeTask("addition with 0.375612",
                 arrow(treal, treal),
                 addFractionHard)
    ]
    """ makeTask("vector addition (3)",
                 arrow(tvector, tvector, tvector, tvector),
                 lambda v1, v2, v3: vectorAddition(v3, vectorAddition(v1, v2))),
        makeTask("vector addition (4)",
                 arrow(tvector, tvector, tvector, tvector, tvector),
                 lambda v1, v2, v3, v4: vectorAddition(v4, vectorAddition(v3, vectorAddition(v1, v2)))),
        makeTask("vector addition (2) + sigmoid",
                 arrow(tvector, tvector, tvector),
                 lambda v1, v2: applySigmoid(vectorAddition(v1, v2))),
        makeTask("vector addition (3) + tanh",
                 arrow(tvector, tvector, tvector, tvector),
                 lambda v1, v2, v3: applyTanh(vectorAddition(v3, vectorAddition(v1, v2)))),
        makeTask("vector multiplication (2)",
                 arrow(tvector, tvector, tvector),
                 vectorMultiplication),
        makeTask("vector multiplication (2) + tanh",
                 arrow(tvector, tvector, tvector),
                 lambda v1, v2: applyTanh(vectorMultiplication(v1, v2))),
        makeTask("vector sum",
                 arrow(tvector, treal),
                 vectorSum),
        makeTask("dot product",
                 arrow(tvector, tvector, treal),
                 dotProduct),
        makeTask("scale vector",
                 arrow(treal, tvector, tvector),
                 scaleVector),
        makeTask("scale vector + sigmoid",
                 arrow(treal, tvector, tvector),
                 lambda r, v: applySigmoid(scaleVector(r, v))),
        #makeTask("sigmoid",
        #         arrow(treal, treal),
        #         _sigmoid),
        makeTask("vector of vector sums",
                 arrow(tvector, tvector, tvector, tvector),
                 lambda v1, v2, v3: [vectorSum(v1), vectorSum(v2), vectorSum(v3)]),
        makeTask("singleton",
                 arrow(treal, tvector),
                 lambda x: [x]),
        makeTask("reals to vector (2) + tanh",
                 arrow(treal, treal, tvector),
                 lambda x1, x2: applyTanh([x1, x2])),
        makeTask("reals to vector (3) + sigmoid",
                 arrow(treal, treal, treal, tvector),
                 lambda x1, x2, x3: applySigmoid([x1, x2, x3])),
        makeTask("adding vector 1 and vector 3",
                 arrow(tvector, tvector, tvector, tvector),
                 lambda v1, v2, v3: vectorAddition(v1, v3)),
        makeTask("adding vector 2 and vector 3 + sigmoid",
                 arrow(tvector, tvector, tvector, tvector),
                 lambda v1, v2, v3: applySigmoid(vectorAddition(v2, v3))),
        makeTask("sum matrix rows", 
                 arrow(tlist(tvector), tvector),
                 sum_matrix_rows, triple=True),
        makeTask("matrix multiplication", 
                 arrow(tlist(tvector), tvector, tvector),
                 _matrix_mult, triple=True),
        makeTask("matrix multiplication + sigmoid", 
                 arrow(tlist(tvector), tvector, tvector),
                 lambda m, l: applySigmoid(_matrix_mult(m, l)), 
                 triple=True),
        #makeTask("matrix mult 3 vectors",
        #         arrow(tvector, tvector, tvector, tvector, tvector),
        #         _matrix_mult_vectors),
        makeTask("matrix mult + addition",
                 arrow(tlist(tvector), tvector, tvector, tvector),
                 lambda m, l, b: vectorAddition(_matrix_mult(m, l), b),
                 triple=True),
        makeTask("matrix mult + addition w scaled vector",
                 arrow(tlist(tvector), tvector, treal, tvector, tvector),
                 lambda m, l, W, b: vectorAddition(_matrix_mult(m, l), scaleVector(W, b)),
                 triple=True),
        makeTask("addition with a scaled vector",
                 arrow(tvector, treal, tvector, tvector),
                 lambda v, W, b: vectorAddition(scaleVector(W, b), v)),
        makeTask("scaled vector addition",
                 arrow(treal, tvector, treal, tvector, tvector),
                 lambda W1, b1, W2, b2: vectorAddition(scaleVector(W1, b1), scaleVector(W2, b2))),
        makeTask("scaled vector addition + tanh",
                 arrow(treal, tvector, treal, tvector, tvector),
                 lambda W1, b1, W2, b2: applyTanh(vectorAddition(scaleVector(W1, b1), scaleVector(W2, b2)))),
        makeTask("forget gate without sigmoid",
                 arrow(tvector, tvector, treal, tlist(tvector), tvector, tvector, tvector),
                 _forget_gate_noActivation, 
                 triple=True),
        makeTask("forget gate",
                 arrow(tvector, tvector, treal, tlist(tvector), tvector, tvector, tvector),
                 _forget_gate,
                 triple=True),
        makeTask("forget gate + vector",
                 arrow(tvector, tvector, treal, tlist(tvector), tvector, tvector, tvector, tvector),
                 lambda htm1, ctm1, xt, Uf, Wf, bf, v: vectorAddition(_forget_gate(htm1, ctm1, xt, Uf, Wf, bf), v),
                 triple=True),
        makeTask("candidate cell state + vector",
                 arrow(tvector, tvector, treal, tlist(tvector), tvector, tvector, tvector, tvector),
                 lambda htm1, ctm1, xt, Uf, Wf, bf, v: vectorAddition(_candidate_cell_state(htm1, ctm1, xt, Uf, Wf, bf), v),
                 triple=True),
        makeTask("candidate cell state * vector",
                 arrow(tvector, tvector, treal, tlist(tvector), tvector, tvector, tvector, tvector),
                 lambda htm1, ctm1, xt, Uf, Wf, bf, v: vectorMultiplication(_candidate_cell_state(htm1, ctm1, xt, Uf, Wf, bf), v),
                 triple=True),
        makeTask("forget gate * vector",
                 arrow(tvector, tvector, treal, tlist(tvector), tvector, tvector, tvector, tvector),
                 lambda htm1, ctm1, xt, Uf, Wf, bf, v: vectorMultiplication(_forget_gate(htm1, ctm1, xt, Uf, Wf, bf), v),
                 triple=True),
        makeTask("forget gate + vector reuse htm1",
                 arrow(tvector, tvector, treal, tlist(tvector), tvector, tvector, tvector),
                 lambda htm1, ctm1, xt, Uf, Wf, bf: vectorAddition(_forget_gate(htm1, ctm1, xt, Uf, Wf, bf), htm1),
                 triple=True),
        makeTask("forget gate + scaled vector reuse xt",
                 arrow(tvector, tvector, treal, tlist(tvector), tvector, tvector, tvector, tvector),
                 lambda htm1, ctm1, xt, Uf, Wf, bf, v: vectorAddition(_forget_gate(htm1, ctm1, xt, Uf, Wf, bf), scaleVector(xt, v)),
                 triple=True),
        makeTask("forget gate + matrix mult",
                 arrow(tvector, tvector, treal, tlist(tvector), tvector, tvector, tlist(tvector), tvector, tvector),
                 lambda htm1, ctm1, xt, Uf, Wf, bf, U, v: vectorAddition(_forget_gate(htm1, ctm1, xt, Uf, Wf, bf), _matrix_mult(U, v)),
                 triple=True),
        makeTask("forget gate + matrix mult reuse htm1 + bias",
                 arrow(tvector, tvector, treal, tlist(tvector), tvector, tvector, tlist(tvector), tvector, tvector),
                 lambda htm1, ctm1, xt, Uf, Wf, bf, U, b: vectorAddition(vectorAddition(_forget_gate(htm1, ctm1, xt, Uf, Wf, bf), _matrix_mult(U, htm1)), b),
                 triple=True),
        makeTask("forget gate addition",
                 arrow(tvector, tvector, treal, tlist(tvector), tvector, tvector, tlist(tvector), tvector, tvector, tvector),
                 lambda htm1, ctm1, xt, Uf, Wf, bf, Ui, Wi, bi: vectorAddition(_forget_gate(htm1, ctm1, xt, Uf, Wf, bf), _forget_gate(htm1, ctm1, xt, Ui, Wi, bi)),
                 triple=True),
        makeTask("forget gate multiplication",
                 arrow(tvector, tvector, treal, tlist(tvector), tvector, tvector, tlist(tvector), tvector, tvector, tvector),
                 lambda htm1, ctm1, xt, Uf, Wf, bf, Ui, Wi, bi: vectorMultiplication(_forget_gate(htm1, ctm1, xt, Uf, Wf, bf), _forget_gate(htm1, ctm1, xt, Ui, Wi, bi)),
                 triple=True),
        makeTask("candidate cell state",
                 arrow(tvector, tvector, treal, tlist(tvector), tvector, tvector, tvector),
                 _candidate_cell_state,
                 triple=True),
        makeTask("forget gate + candidate cell state",
                 arrow(tvector, tvector, treal, tlist(tvector), tvector, tvector, tlist(tvector), tvector, tvector, tvector),
                 lambda htm1, ctm1, xt, Uf, Wf, bf, Uc, Wc, bc: vectorAddition(_forget_gate(htm1, ctm1, xt, Uf, Wf, bf), _candidate_cell_state(htm1, ctm1, xt, Uc, Wc, bc)),
                 triple=True),
        makeTask("vector sum of 2 vector multiplications",
                 arrow(tvector, tvector, tvector, tvector, tvector),
                 _new_cell_state),
        makeTask("new hidden state",
                 arrow(tvector, tvector, tvector),
                 _new_hidden_state),
        makeTask("lstm cell state no activations",
                 arrow(tvector, tvector, treal, tlist(tvector), tvector, tvector, tlist(tvector), tvector, tvector, tlist(tvector), tvector, tvector, tvector),
                 _lstm_cell_state_noActivation,
                 triple=True),
        makeTask("lstm cell state",
                 arrow(tvector, tvector, treal, tlist(tvector), tvector, tvector, tlist(tvector), tvector, tvector, tlist(tvector), tvector, tvector, tvector),
                 _lstm_cell_state,
                 triple=True), 
        makeTask("lstm cell",
                 arrow(tvector, tvector, treal, tlist(tvector), tvector, tvector, tlist(tvector), tvector, tvector, tlist(tvector), tvector, tvector, tlist(tvector), tvector, tvector, tlist(tvector)),
                 _lstm_cell,
                 triple=True) """
    #]

    real = Primitive("REAL", treal, None)
    bootstrapTarget()
    equationPrimitives = [
        real,
        f0,
        f1,
        #real_power,
        #real_subtraction,
        real_addition,
        real_division,
        real_multiplication,
        prim_tanh,
        prim_sigmoid] + [
        #prim_relu] + [
            Program.parse(n)
            for n in ["map","fold",
                      "empty","cons","car","cdr",
                      "zip", "reduce"]]
    baseGrammar = Grammar.uniform(equationPrimitives)

    eprint("Got %d equation discovery tasks..." % len(tasks))

    explorationCompression(baseGrammar, tasks,
                           outputPrefix="experimentOutputs/lstm",
                           evaluationTimeout=0.1,
                           testingTasks=[],
                           **commandlineArguments(
                               compressor="ocaml",
                               featureExtractor=LearnedFeatureExtractor,
                               iterations=10,
                               CPUs=numberOfCPUs(),
                               structurePenalty=0.5,
                               helmholtzRatio=0.5,
                               a=3,
                               maximumFrontier=10000,
                               topK=2,
                               pseudoCounts=10.0))
