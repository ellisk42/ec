from ec import explorationCompression, commandlineArguments, Task, ecIterator
from grammar import Grammar
from utilities import eprint, testTrainSplit, numberOfCPUs, median, standardDeviation, mean
from makeTextTasks import makeTasks, delimiters, loadPBETasks
from textPrimitives import primitives, targetTextPrimitives
from listPrimitives import bootstrapTarget
from program import *
from recognition import *
from enumeration import *

import os
import datetime
import random
from functools import reduce
import dill


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
    special = 'string'
    
    def tokenize(self, examples):
        return examples

    def __init__(self, tasks, testingTasks=[], cuda=False):
        lexicon = {c
                   for t in tasks + testingTasks
                   for xs, y in self.tokenize(t.examples)
                   for c in reduce(lambda u, v: u + v, list(xs) + [y])}

        self.recomputeTasks = True

        super(LearnedFeatureExtractor, self).__init__(lexicon=list(lexicon),
                                                      H=64,
                                                      tasks=tasks,
                                                      bidirectional=True,
                                                      cuda=cuda)
        self.MAXINPUTS = 8

    def taskOfProgram(self, p, tp):
        # Instantiate STRING w/ random words
        p = p.visit(ConstantInstantiateVisitor.SINGLE)
        return super(LearnedFeatureExtractor, self).taskOfProgram(p, tp)


### COMPETITION CODE

def competeOnOneTask(checkpoint, task,
                     CPUs=8, timeout=3600, evaluationTimeout=0.0005):
    if checkpoint.recognitionModel is not None:
        recognizer = checkpoint.recognitionModel
        challengeFrontiers, times, bestSearchTime = \
                recognizer.enumerateFrontiers([task], "all-or-nothing",
                                              CPUs=CPUs,
                                              solver="ocaml",
                                              maximumFrontier=1,
                                              enumerationTimeout=timeout,
                                              evaluationTimeout=evaluationTimeout)
    else:
        challengeFrontiers, times, bestSearchTimes = \
                multicoreEnumeration(checkpoint.grammars[-1], [task], "all-or-nothing",
                                     CPUs=CPUs,
                                     solver="ocaml",
                                     maximumFrontier=1,
                                     enumerationTimeout=timeout,
                                     evaluationTimeout=evaluationTimeout)
    if len(times) == 0: return None, task
    assert len(times) == 1
    return times[0], task

        

def sygusCompetition(checkpoints, tasks):
    from pathos.multiprocessing import Pool
    import datetime
    
    # map from task to list of search times, one for each checkpoint.
    # search time will be None if it is not solved
    searchTimes = {t: [] for t in tasks}

    CPUs = int(8/len(checkpoints))
    eprint(f"You gave me {len(checkpoints)} checkpoints to ensemble. Each checkpoint will get {CPUs} CPUs.")
    timeout = 3600

    maxWorkers = int(numberOfCPUs()/CPUs)
    workers = Pool(maxWorkers)

    promises = []
    for t in tasks:
        for checkpoint in checkpoints:
            promise = workers.apply_async(competeOnOneTask,
                                          (checkpoint,t),
                                          {"CPUs": CPUs,
                                           "timeout": timeout})
            promises.append(promise)
    for promise in zip(promises, tasks):
        dt, task = promise.get()
        if dt is not None:
            searchTimes[task].append(dt)

    searchTimes = {t: min(ts) if len(ts) > 0 else None
                   for t,ts in searchTimes.items()}
    
    fn = "experimentOutputs/text_competition_%s.p"%(datetime.datetime.now().isoformat())
    with open(fn,"wb") as handle:
        pickle.dump(searchTimes, handle)
    eprint()

    hits = sum( t is not None for t in searchTimes.values() )
    total = len(searchTimes)
    percentage = 100*hits/total
    eprint("Hits %d/%d = %f\n"%(hits, total, percentage))
    eprint()
    eprint("Exported competition results to",fn)
    
    

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
    parser.add_argument(
        "--compete",
        nargs='+',
        default=None,
        type=str,
        help="Do a simulated sygus competition (1hr+8cpus/problem) on the sygus tasks, restoring from provided checkpoint(s). If multiple checkpoints are provided, then we ensemble the models.")

if __name__ == "__main__":
    arguments = commandlineArguments(
        recognitionTimeout=7200,
        iterations=10,
        helmholtzRatio=0.5,
        topK=2,
        maximumFrontier=5,
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
        ConstantInstantiateVisitor(list(map(list, list({tuple([c for c in s])
                                                        for t in test + train + challenge
                                                        for s in t.stringConstants}))))

    baseGrammar = Grammar.uniform(primitives + bootstrapTarget())
    challengeGrammar = baseGrammar  # Grammar.uniform(targetTextPrimitives)

    evaluationTimeout = 0.0005
    # We will spend 10 minutes on each challenge problem
    challengeTimeout = 10 * 60

    for t in train + test + challenge:
        t.maxParameters = 1

    competitionCheckpoints = arguments.pop("compete")
    if competitionCheckpoints:
        checkpoints = []
        for competitionCheckpoint in competitionCheckpoints:
            with open(competitionCheckpoint, 'rb') as handle:
                checkpoints.append(dill.load(handle))
        sygusCompetition(checkpoints, challenge)
        sys.exit(0)

    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/text/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)


    generator = ecIterator(baseGrammar, train,
                           testingTasks=test + challenge,
                           outputPrefix="%s/text"%outputDirectory,
                           evaluationTimeout=evaluationTimeout,
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
