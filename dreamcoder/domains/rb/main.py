from dreamcoder.dreamcoder import ecIterator
from dreamcoder.domains.text.makeTextTasks import makeTasks, loadPBETasks
from dreamcoder.domains.text.textPrimitives import primitives
from dreamcoder.domains.list.listPrimitives import bootstrapTarget
from dreamcoder.recognition import *
from dreamcoder.enumeration import *

import os
import datetime
import random
from functools import reduce
import dill



"""
TODO:
- [ ] 
- [ ] import robut and use its sampling
- [ ] can use robut execution I suppose, or can use own execution

- [ ] implement dsl

- [ ] value eval code (seperate possibly?)
- [ ] neural net via featureExtractor

- [ ] helmholtz stuff

- [ ] make test tasks


"""


class LearnedFeatureExtractor(RecurrentFeatureExtractor):
    special = 'string'
    
    def tokenize(self, examples):
        def tokenize_example(xs,y):
            if not isinstance(y, list): y = [y]
            return xs,y
        return [tokenize_example(*e) for e in examples]

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
    

def text_options(parser):
    parser.add_argument(
        "--showTasks",
        action="store_true",
        default=False,
        help="show the training test and challenge tasks and then exit")
    parser.add_argument(
        "--trainChallenge",
        action="store_true",
        default=False,
        help="Incorporate a random 50% of the challenge problems into the training set")
    parser.add_argument(
        "--onlyChallenge",
        action="store_true",
        default=False,
        help="Only train on challenge problems and have testing problems.")
    parser.add_argument(
        "--latest",
        action="store_true",
        default=False,
        help="evaluate on latest sygus problems rather than problems used in ec2 paper")
    parser.add_argument(
        "--noMap", action="store_true", default=False,
        help="Disable built-in map primitive")
    parser.add_argument(
        "--noLength", action="store_true", default=False,
        help="Disable built-in length primitive")
    parser.add_argument(
        "--noUnfold", action="store_true", default=False,
        help="Disable built-in unfold primitive")
    parser.add_argument(
        "--compete",
        nargs='+',
        default=None,
        type=str,
        help="Do a simulated sygus competition (1hr+8cpus/problem) on the sygus tasks, restoring from provided checkpoint(s). If multiple checkpoints are provided, then we ensemble the models.")


def main(arguments):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on manipulating sequences of text.
    """

    tasks = makeTasks()
    eprint("Generated", len(tasks), "tasks")

    for t in tasks:
        t.mustTrain = False

    test, train = testTrainSplit(tasks, 1.)
    eprint("Split tasks into %d/%d test/train" % (len(test), len(train)))

    latest = arguments.pop("latest")
    challenge, challengeCheating = loadPBETasks("data/sygus" if latest else "PBE_Strings_Track")
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

    if arguments.pop('onlyChallenge'):
        train = challenge
        test = []
        challenge = []
        eprint("Training only on sygus problems.")
        

    ConstantInstantiateVisitor.SINGLE = \
        ConstantInstantiateVisitor(list(map(list, list({tuple([c for c in s])
                                                        for t in test + train + challenge
                                                        for s in t.stringConstants}))))

    haveLength = not arguments.pop("noLength")
    haveMap = not arguments.pop("noMap")
    haveUnfold = not arguments.pop("noUnfold")
    eprint(f"Including map as a primitive? {haveMap}")
    eprint(f"Including length as a primitive? {haveLength}")
    eprint(f"Including unfold as a primitive? {haveUnfold}")
    baseGrammar = Grammar.uniform(primitives + [p
                                                for p in bootstrapTarget()
                                                if (p.name != "map" or haveMap) and \
                                                (p.name != "unfold" or haveUnfold) and \
                                                (p.name != "length" or haveLength)])
    challengeGrammar = baseGrammar  # Grammar.uniform(targetTextPrimitives)

    evaluationTimeout = 0.0005
    # We will spend 10 minutes on each challenge problem
    challengeTimeout = 10 * 60

    for t in train + test + challenge:
        t.maxParameters = 2

    if arguments.pop("showTasks"):
        for source, ts in [("train",tasks),("test",test),("challenge",challenge)]:
            print(source,"tasks:")
            for t in ts:
                print(t.name)
                for xs, y in t.examples:
                    xs = ['"' + "".join(x) + '"' for x in xs]
                    y = "".join(y) if isinstance(y,list) else y
                    print('f(%s) = "%s"' % (", ".join(xs), y))
                print("\t{%s}" % (t.stringConstants))
            print()
        sys.exit(0)


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
    for result in generator:
        pass
