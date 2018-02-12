import cPickle as pickle
import random
from collections import defaultdict
from itertools import chain
from ec import explorationCompression, commandlineArguments
from utilities import eprint, numberOfCPUs, flatten, fst, testTrainSplit
from grammar import Grammar
from task import Task
from type import Context, arrow, tlist, tint, t0, UnificationFailure
from listPrimitives import basePrimitives, primitives
from recognition import HandCodedFeatureExtractor, MLPFeatureExtractor, RecurrentFeatureExtractor
from enumeration import enumerateForTask

def retrieveTasks(filename):
    with open(filename) as f:
        return pickle.load(f)


def list_features(examples):
    if any(isinstance(i, int) for (i,), _ in examples):
        # obtain features for number inputs as list of numbers
        examples = [(([i],), o) for (i,), o in examples]
    elif any(not isinstance(i, list) for (i,), _ in examples):
        # can't handle non-lists
        return []
    elif any(isinstance(x, list) for (xs,), _ in examples for x in xs):
        # nested lists are hard to extract features for, so we'll
        # obtain features as if flattened
        examples = [(([x for xs in ys for x in xs],), o) for (ys,), o in examples]

    # assume all tasks have the same number of examples
    # and all inputs are lists
    features = []
    ot = type(examples[0][1])
    mean = lambda l: 0 if not l else sum(l)/len(l)
    imean = [mean(i) for (i,), o in examples]
    ivar = [sum((v - imean[idx])**2
                for v in examples[idx][0][0])
            for idx in xrange(len(examples))]

    #DISABLED length of each input and output
    # total difference between length of input and output
    #DISABLED normalized count of numbers in input but not in output
    # total normalized count of numbers in input but not in output
    # total difference between means of input and output
    # total difference between variances of input and output
    # output type (-1=bool, 0=int, 1=list)
    #DISABLED outputs if integers, else -1s
    #DISABLED outputs if bools (-1/1), else 0s
    if ot == list:  # lists of ints or bools
        omean = [mean(o) for (i,), o in examples]
        ovar = [sum((v - omean[idx])**2
                    for v in examples[idx][1])
                for idx in xrange(len(examples))]
        cntr = lambda l, o: 0 if not l else len(set(l).difference(set(o))) / len(l)
        cnt_not_in_output = [cntr(i, o) for (i,), o in examples]

        #features += [len(i) for (i,), o in examples]
        #features += [len(o) for (i,), o in examples]
        features.append(sum(len(i) - len(o) for (i,), o in examples))
        #features += cnt_not_int_output
        features.append(sum(cnt_not_in_output))
        features.append(sum(om - im for im, om in zip(imean, omean)))
        features.append(sum(ov - iv for iv, ov in zip(ivar, ovar)))
        features.append(1)
        # features += [-1 for _ in examples]
        # features += [0 for _ in examples]
    elif ot == bool:
        outs = [o for (i,), o in examples]

        #features += [len(i) for (i,), o in examples]
        #features += [-1 for _ in examples]
        features.append(sum(len(i) for (i,), o in examples))
        #features += [0 for _ in examples]
        features.append(0)
        features.append(sum(imean))
        features.append(sum(ivar))
        features.append(-1)
        # features += [-1 for _ in examples]
        # features += [1 if o else -1 for o in outs]
    else:  # int
        cntr = lambda l, o: 0 if not l else len(set(l).difference(set(o))) / len(l)
        cnt_not_in_output = [cntr(i, [o]) for (i,), o in examples]
        outs = [o for (i,), o in examples]

        #features += [len(i) for (i,), o in examples]
        #features += [1 for (i,), o in examples]
        features.append(sum(len(i) for (i,), o in examples))
        #features += cnt_not_int_output
        features.append(sum(cnt_not_in_output))
        features.append(sum(o - im for im, o in zip(imean, outs)))
        features.append(sum(ivar))
        features.append(0)
        # features += outs
        # features += [0 for _ in examples]

    return features


def isListFunction(tp):
    try:
        Context().unify(tp, arrow(tlist(tint), t0))
        return True
    except UnificationFailure:
        return False


def isIntFunction(tp):
    try:
        Context().unify(tp, arrow(tint, t0))
        return True
    except UnificationFailure:
        return False


class FeatureExtractor(HandCodedFeatureExtractor):
    N_EXAMPLES = 15
    def _featuresOfProgram(self, program, tp):
        e = program.evaluate([])
        examples = []
        if isListFunction(tp):
            sample = lambda: random.sample(xrange(30), random.randint(0, 8))
        elif isIntFunction(tp):
            sample = lambda: random.randint(0, 20)
        else: return None
        for _ in xrange(self.N_EXAMPLES*5):
            x = sample()
            try:
                y = e(x)
                examples.append(((x,), y))
            except: continue
            if len(examples) >= self.N_EXAMPLES: break
        else: return None
        return list_features(examples)


class DeepFeatureExtractor(MLPFeatureExtractor):
    N_EXAMPLES = 15
    H = 16
    USE_CUDA = False
    def __init__(self, tasks):
        super(DeepFeatureExtractor, self).__init__(tasks, cuda=self.USE_CUDA, H=self.H)
    def _featuresOfProgram(self, program, tp):
        e = program.evaluate([])
        examples = []
        if isListFunction(tp):
            sample = lambda: random.sample(xrange(30), random.randint(0, 8))
        elif isIntFunction(tp):
            sample = lambda: random.randint(0, 20)
        else: return None
        for _ in xrange(self.N_EXAMPLES*5):
            x = sample()
            try:
                y = e(x)
                examples.append(((x,), y))
            except: continue
            if len(examples) >= self.N_EXAMPLES: break
        else: return None
        return list_features(examples)


class LearnedFeatureExtractor(RecurrentFeatureExtractor):
    H = 16
    USE_CUDA = False
    def __init__(self, tasks):
        def tokenize(examples, lexicon):
            tokenized = []
            for (x,), y in examples:
                if isinstance(x, list):
                    x = ["LIST_START"]+x+["LIST_END"]
                else:
                    x = [x]
                if isinstance(y, list):
                    y = ["LIST_START"]+y+["LIST_END"]
                else:
                    y = [y]
                if True:  # force successful tokenization
                    x = [e if e in lexicon else "?" for e in x]
                    y = [e if e in lexicon else "?" for e in y]
                    if len(x) > 25 or len(y) > 25:
                        continue
                    tokenized.append(((x,), y))
                else:
                    if all(e in lexicon for e in chain(x, y)):
                        tokenized.append(((x,), y))
                    else:
                        eprint("could not tokenize {} because {} not in lexicon".format(
                            ((x,), y), fst(e for e in chain(x, y) if e not in lexicon)))
            return tokenized
        lexicon = set(flatten((t.examples for t in tasks), abort=lambda x:isinstance(x, str))).union({"LIST_START", "LIST_END", "?"})
        super(LearnedFeatureExtractor, self).__init__(lexicon=list(lexicon),
                                                      tasks=tasks,
                                                      cuda=self.USE_CUDA,
                                                      H=self.H,
                                                      bidirectional=True,
                                                      tokenize=tokenize)


def train_necessary(task):
    if t.name in {"head", "is-primes", "len", "pop", "repeat-many", "tail"}:
        return True
    if any(t.name.startswith(x) for x in {
            "add-k", "append-k", "bool-identify-geq-k", "count-k", "drop-k",
            "empty", "evens", "has-k", "index-k", "is-mod-k", "kth-largest",
            "kth-smallest", "modulo-k", "mult-k", "remove-index-k",
            "remove-mod-k", "repeat-k", "replace-all-with-index-k", "rotate-k",
            "slice-k-n", "take-k",
        }):
        return "some"
    return False


def list_options(parser):
    parser.add_argument("--dataset", type=str,
        default="data/list_tasks.pkl",
        help="location of pickled list function dataset")
    parser.add_argument("--maxTasks", type=int,
        default=1000,
        help="truncate tasks to fit within this boundary")
    parser.add_argument("--base", action="store_true",
        help="use foundational primitives")
    parser.add_argument("--extractor", type=str,
        choices=["hand", "deep", "learned"],
        default="learned")
    parser.add_argument("--split", metavar="TRAIN_RATIO",
        type=float,
        help="split test/train")
    parser.add_argument("-H", "--hidden", type=int,
        default=16,
        help="number of hidden units")


if __name__ == "__main__":
    args = commandlineArguments(
        frontierSize=None, activation='sigmoid', iterations=10,
        a=3, maximumFrontier=10, topK=2, pseudoCounts=10.0,
        helmholtzRatio=0.5, structurePenalty=1.,
        CPUs=numberOfCPUs(),
        extras=list_options)

    tasks = retrieveTasks(args.pop("dataset"))

    maxTasks = args.pop("maxTasks")
    if len(tasks) > maxTasks:
        eprint("Unwilling to handle {} tasks, truncating..".format(len(tasks)))
        random.seed(42)
        random.shuffle(tasks)
        del tasks[maxTasks:]

    eprint("Got {} list tasks".format(len(tasks)))

    for task in tasks:
        task.features = list_features(task.examples)
        task.cache = False

    split = args.pop("split")
    if split:
        random.seed(42)
        train = []
        train_some = defaultdict(list)
        for t in tasks:
            necessary = train_necessary(t)
            if not necessary:
                continue
            if necessary == "some":
                train_some[t.name.split()[0]].append(t)
            else:
                train.append(t)
        for k in sorted(train_some):
            ts = train_some[k]
            random.shuffle(ts)
            train.append(ts.pop())

        tasks = [t for t in tasks if t not in train]
        test, more_train = testTrainSplit(tasks, split)
        train.extend(more_train)

        eprint("Alotted {} tasks for training and {} for testing".format(len(train), len(test)))
    else:
        train = tasks
        test = []

    prims = basePrimitives if args.pop("base") else primitives

    extractor = {
        "hand": FeatureExtractor,
        "deep": DeepFeatureExtractor,
        "learned": LearnedFeatureExtractor,
    }[args.pop("extractor")]
    extractor.H = args.pop("hidden")
    extractor.USE_CUDA = args["cuda"]

    args.update({
        "featureExtractor": extractor,
        "outputPrefix": "experimentOutputs/list",
    })

    baseGrammar = Grammar.uniform(prims())
    from program import *
    p = Program.parse("(lambda (lambda (if (is-square $0) (++ (singleton $0) $1) $1)))")
    eprint(p)
    eprint(baseGrammar.closedLogLikelihood(arrow(tlist(tint),tint,tlist(tint)), p))
    assert False
    explorationCompression(baseGrammar, train, testingTasks=test, **args)
