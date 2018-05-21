from ec import explorationCompression, commandlineArguments, Task, ecIterator
from grammar import Grammar
from utilities import eprint, testTrainSplit, numberOfCPUs, median, standardDeviation, mean
from makeTextTasks import makeTasks, delimiters, loadPBETasks
from textPrimitives import primitives, targetTextPrimitives
from listPrimitives import bootstrapTarget
from program import *
from recognition import *
from enumeration import *

import random
from functools import reduce


class ConstantInstantiateVisitor(object):
    def __init__(self, words):
        self.words = words

    def primitive(self, e):
        if e.name == "STRING":
            return Primitive("STRING", e.tp, random.choice(self.words))
        return e

    def invented(self, e): return e.body.visit(self)

    def index(self, e): return e

    def application(self, e):
        return Application(e.f.visit(self), e.x.visit(self))

    def abstraction(self, e):
        return Abstraction(e.body.visit(self))


class LearnedFeatureExtractor(RecurrentFeatureExtractor):
    def tokenize(self, examples):
        return examples

    def __init__(self, tasks):
        lexicon = {c
                   for t in tasks
                   for xs, y in self.tokenize(t.examples)
                   for c in reduce(lambda u, v: u + v, list(xs) + [y])}

        super(LearnedFeatureExtractor, self).__init__(lexicon=list(lexicon),
                                                      H=64,
                                                      tasks=tasks,
                                                      bidirectional=True)
        self.MAXINPUTS = 4

    def taskOfProgram(self, p, tp):
        # Instantiate STRING w/ random words
        p = p.visit(ConstantInstantiateVisitor.SINGLE)
        return super(LearnedFeatureExtractor, self).taskOfProgram(p, tp)


def text_options(parser):
    parser.add_argument(
        "--doChallenge",
        action="store_true",
        default=False,
        help="Evaluate model after/before each iteration on sygus")
    parser.add_argument(
        "--trainChallenge",
        action="store_true",
        default=False,
        help="Incorporate a random 50% of the challenge problems into the training set")


if __name__ == "__main__":
    arguments = commandlineArguments(
        steps=250,
        iterations=10,
        helmholtzRatio=0.5,
        topK=2,
        maximumFrontier=2,
        structurePenalty=10.,
        a=3,
        activation="tanh",
        CPUs=numberOfCPUs(),
        featureExtractor=LearnedFeatureExtractor,
        pseudoCounts=30.0,
        extras=text_options)
    doChallenge = arguments.pop('doChallenge')

    tasks = makeTasks()
    eprint("Generated", len(tasks), "tasks")

    for t in tasks:
        t.mustTrain = False

    test, train = testTrainSplit(tasks, 1.)
    eprint("Split tasks into %d/%d test/train" % (len(test), len(train)))

    challenge, challengeCheating = loadPBETasks()
    eprint("Got %d challenge PBE tasks" % len(challenge))

    if arguments.pop('trainChallenge'):
        challengeTest, challengeTrain = testTrainSplit(challenge, 0.5)
        challenge = challengeTest
        train += challengeTrain
        eprint(
            "Incorporating %d (50%%) challenge problems into the training set." %
            (len(challengeTrain)),
            "We will evaluate on the held out challenge problems.",
            "This makes a total of %d training problems." %
            len(train))

    ConstantInstantiateVisitor.SINGLE = \
        ConstantInstantiateVisitor(map(list, list({tuple([c for c in s])
                                                   for t in test + train + challenge
                                                   for s in t.stringConstants})))

    baseGrammar = Grammar.uniform(primitives + bootstrapTarget())
    challengeGrammar = baseGrammar  # Grammar.uniform(targetTextPrimitives)

    evaluationTimeout = 0.0005
    # We will spend 10 minutes on each challenge problem
    challengeTimeout = 10 * 60

    generator = ecIterator(baseGrammar, train,
                           testingTasks=test + challenge,
                           outputPrefix="experimentOutputs/text",
                           evaluationTimeout=evaluationTimeout,
                           compressor="rust",
                           **arguments)
    if doChallenge:
        eprint("held out challenge problems before learning...")
        challengeFrontiers, times = multicoreEnumeration(challengeGrammar, challenge, "all-or-nothing",
                                                         CPUs=numberOfCPUs(),
                                                         solver="ocaml",
                                                         maximumFrontier=1,
                                                         enumerationTimeout=challengeTimeout,
                                                         evaluationTimeout=evaluationTimeout)
        eprint(Frontier.describe(challengeFrontiers))
        summaryStatistics("Challenge problem search time", times)
        eprint("done evaluating challenge problems before learning")

    for result in generator:
        if not doChallenge:
            continue

        eprint("Evaluating on challenge problems...")
        if result.recognitionModel is not None:
            recognizer = result.recognitionModel
            challengeFrontiers, times = \
                recognizer.enumerateFrontiers(challenge, "all-or-nothing",
                                              CPUs=numberOfCPUs(),
                                              solver="ocaml",
                                              maximumFrontier=1,
                                              enumerationTimeout=challengeTimeout,
                                              evaluationTimeout=evaluationTimeout)
        else:
            challengeFrontiers, times = \
                multicoreEnumeration(result.grammars[-1], challenge, "all-or-nothing",
                                     CPUs=numberOfCPUs(),
                                     solver="ocaml",
                                     maximumFrontier=1,
                                     enumerationTimeout=challengeTimeout,
                                     evaluationTimeout=evaluationTimeout)
        eprint("Challenge problem enumeration results:")
        eprint(Frontier.describe(challengeFrontiers))
        summaryStatistics("Challenge problem search time", times)
