import pickle as pickle
import random
from collections import defaultdict
from itertools import chain
import json
import math

from ec import explorationCompression, commandlineArguments
from utilities import eprint, numberOfCPUs, flatten, fst, testTrainSplit, POSITIVEINFINITY
from grammar import Grammar
from task import Task
from type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
from listPrimitives import basePrimitives, primitives, McCarthyPrimitives, bootstrapTarget_extra, no_length
from recognition import HandCodedFeatureExtractor, MLPFeatureExtractor, RecurrentFeatureExtractor
from makeListTasks import make_list_bootstrap_tasks, sortBootstrap


def retrieveJSONTasks(filename, features=False):
    """
    For JSON of the form:
        {"name": str,
         "type": {"input" : bool|int|list-of-bool|list-of-int,
                  "output": bool|int|list-of-bool|list-of-int},
         "examples": [{"i": data, "o": data}]}
    """
    with open(filename, "r") as f:
        loaded = json.load(f)
    TP = {
        "bool": tbool,
        "int": tint,
        "list-of-bool": tlist(tbool),
        "list-of-int": tlist(tint),
    }
    return [Task(
        item["name"],
        arrow(TP[item["type"]["input"]], TP[item["type"]["output"]]),
        [((ex["i"],), ex["o"]) for ex in item["examples"]],
        features=(None if not features else list_features(
            [((ex["i"],), ex["o"]) for ex in item["examples"]])),
        cache=False,
    ) for item in loaded]


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
        examples = [(([x for xs in ys for x in xs],), o)
                    for (ys,), o in examples]

    # assume all tasks have the same number of examples
    # and all inputs are lists
    features = []
    ot = type(examples[0][1])

    def mean(l): return 0 if not l else sum(l) / len(l)
    imean = [mean(i) for (i,), o in examples]
    ivar = [sum((v - imean[idx])**2
                for v in examples[idx][0][0])
            for idx in range(len(examples))]

    # DISABLED length of each input and output
    # total difference between length of input and output
    # DISABLED normalized count of numbers in input but not in output
    # total normalized count of numbers in input but not in output
    # total difference between means of input and output
    # total difference between variances of input and output
    # output type (-1=bool, 0=int, 1=list)
    # DISABLED outputs if integers, else -1s
    # DISABLED outputs if bools (-1/1), else 0s
    if ot == list:  # lists of ints or bools
        omean = [mean(o) for (i,), o in examples]
        ovar = [sum((v - omean[idx])**2
                    for v in examples[idx][1])
                for idx in range(len(examples))]

        def cntr(
            l, o): return 0 if not l else len(
            set(l).difference(
                set(o))) / len(l)
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
        def cntr(
            l, o): return 0 if not l else len(
            set(l).difference(
                set(o))) / len(l)
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
            def sample(): return random.sample(range(30), random.randint(0, 8))
        elif isIntFunction(tp):
            def sample(): return random.randint(0, 20)
        else:
            return None
        for _ in range(self.N_EXAMPLES * 5):
            x = sample()
            try:
                y = e(x)
                examples.append(((x,), y))
            except BaseException:
                continue
            if len(examples) >= self.N_EXAMPLES:
                break
        else:
            return None
        return list_features(examples)


class DeepFeatureExtractor(MLPFeatureExtractor):
    N_EXAMPLES = 15
    H = 16
    USE_CUDA = False

    def __init__(self, tasks):
        super(
            DeepFeatureExtractor,
            self).__init__(
            tasks,
            cuda=self.USE_CUDA,
            H=self.H)

    def _featuresOfProgram(self, program, tp):
        e = program.evaluate([])
        examples = []
        if isListFunction(tp):
            def sample(): return random.sample(range(30), random.randint(0, 8))
        elif isIntFunction(tp):
            def sample(): return random.randint(0, 20)
        else:
            return None
        for _ in range(self.N_EXAMPLES * 5):
            x = sample()
            try:
                y = e(x)
                examples.append(((x,), y))
            except BaseException:
                continue
            if len(examples) >= self.N_EXAMPLES:
                break
        else:
            return None
        return list_features(examples)


class LearnedFeatureExtractor(RecurrentFeatureExtractor):
    H = 16
    USE_CUDA = False

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

    def __init__(self, tasks):
        self.lexicon = set(flatten((t.examples for t in tasks), abort=lambda x: isinstance(
            x, str))).union({"LIST_START", "LIST_END", "?"})

        # Calculate the maximum length
        self.maximumLength = POSITIVEINFINITY
        self.maximumLength = max(len(l)
                                 for t in tasks
                                 for xs, y in self.tokenize(t.examples)
                                 for l in [y] + [x for x in xs])

        super(
            LearnedFeatureExtractor,
            self).__init__(
            lexicon=list(
                self.lexicon),
            tasks=tasks,
            cuda=self.USE_CUDA,
            H=self.H,
            bidirectional=True)


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
    parser.add_argument(
        "--dataset",
        type=str,
        default="Lucas-old",
        choices=[
            "bootstrap",
            "sorting",
            "Lucas-old",
            "Lucas-depth1",
            "Lucas-depth2",
            "Lucas-depth3"])
    parser.add_argument("--maxTasks", type=int,
                        default=None,
                        help="truncate tasks to fit within this boundary")
    parser.add_argument("--primitives",
                        default="common",
                        help="Which primitive set to use",
                        choices=["McCarthy", "base", "rich", "common", "noLength"])
    parser.add_argument("--extractor", type=str,
                        choices=["hand", "deep", "learned"],
                        default="learned")
    parser.add_argument("--split", metavar="TRAIN_RATIO",
                        type=float,
                        help="split test/train")
    parser.add_argument("-H", "--hidden", type=int,
                        default=64,
                        help="number of hidden units")
    parser.add_argument("--random-seed", type=int, default=17)


if __name__ == "__main__":
    args = commandlineArguments(
        enumerationTimeout=10, activation='tanh', iterations=10,
        a=3, maximumFrontier=10, topK=2, pseudoCounts=10.0,
        helmholtzRatio=0.5, structurePenalty=2.,
        CPUs=numberOfCPUs(),
        extras=list_options)

    random.seed(args.pop("random_seed"))

    dataset = args.pop("dataset")
    tasks = {
        "Lucas-old": lambda: retrieveJSONTasks("data/list_tasks.json"),
        "bootstrap": make_list_bootstrap_tasks,
        "sorting": sortBootstrap,
        "Lucas-depth1": lambda: retrieveJSONTasks("data/list_tasks2.json")[:105],
        "Lucas-depth2": lambda: retrieveJSONTasks("data/list_tasks2.json")[:4928],
        "Lucas-depth3": lambda: retrieveJSONTasks("data/list_tasks2.json"),
    }[dataset]()

    maxTasks = args.pop("maxTasks")
    if maxTasks and len(tasks) > maxTasks:
        necessaryTasks = []  # maxTasks will not consider these
        if dataset.startswith("Lucas2.0") and dataset != "Lucas2.0-depth1":
            necessaryTasks = tasks[:105]

        eprint("Unwilling to handle {} tasks, truncating..".format(len(tasks)))
        random.shuffle(tasks)
        del tasks[maxTasks:]
        tasks = necessaryTasks + tasks

    if dataset.startswith("Lucas"):
        # extra tasks for filter
        tasks.extend([
            Task("remove empty lists",
                 arrow(tlist(tlist(tbool)), tlist(tlist(tbool))),
                 [((ls,), list(filter(lambda l: len(l) > 0, ls)))
                  for _ in range(15)
                  for ls in [[[random.random() < 0.5 for _ in range(random.randint(0, 3))]
                              for _ in range(4)]]]),
            Task("keep squares",
                 arrow(tlist(tint), tlist(tint)),
                 [((xs,), list(filter(lambda x: int(math.sqrt(x)) ** 2 == x,
                                      xs)))
                  for _ in range(15)
                  for xs in [[random.choice([0, 1, 4, 9, 16, 25])
                              if random.random() < 0.5
                              else random.randint(0, 9)
                              for _ in range(7)]]]),
            Task("keep primes",
                 arrow(tlist(tint), tlist(tint)),
                 [((xs,), list(filter(lambda x: x in {2, 3, 5, 7, 11, 13, 17,
                                                      19, 23, 29, 31, 37}, xs)))
                  for _ in range(15)
                  for xs in [[random.choice([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])
                              if random.random() < 0.5
                              else random.randint(0, 9)
                              for _ in range(7)]]]),
        ])
        for i in range(4):
            tasks.extend([
                Task("keep eq %s" % i,
                     arrow(tlist(tint), tlist(tint)),
                     [((xs,), list(filter(lambda x: x == i, xs)))
                      for _ in range(15)
                      for xs in [[random.randint(0, 6) for _ in range(5)]]]),
                Task("remove eq %s" % i,
                     arrow(tlist(tint), tlist(tint)),
                     [((xs,), list(filter(lambda x: x != i, xs)))
                      for _ in range(15)
                      for xs in [[random.randint(0, 6) for _ in range(5)]]]),
                Task("keep gt %s" % i,
                     arrow(tlist(tint), tlist(tint)),
                     [((xs,), list(filter(lambda x: x > i, xs)))
                      for _ in range(15)
                      for xs in [[random.randint(0, 6) for _ in range(5)]]]),
                Task("remove gt %s" % i,
                     arrow(tlist(tint), tlist(tint)),
                     [((xs,), list(filter(lambda x: not x > i, xs)))
                      for _ in range(15)
                      for xs in [[random.randint(0, 6) for _ in range(5)]]])
            ])

    prims = {"base": basePrimitives,
             "McCarthy": McCarthyPrimitives,
             "common": bootstrapTarget_extra,
             "noLength": no_length,
             "rich": primitives}[args.pop("primitives")]()
    baseGrammar = Grammar.uniform(prims)

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
        "evaluationTimeout": 0.0005,
        "solver": "ocaml",
        "compressor": "beta"
    })
    

    eprint("Got {} list tasks".format(len(tasks)))
    split = args.pop("split")
    if split:
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

        eprint(
            "Alotted {} tasks for training and {} for testing".format(
                len(train), len(test)))
    else:
        train = tasks
        test = []

    if False:
        from program import *
        p = Program.parse("(lambda (#(lambda (cdr (cdr $0))) (fold (map (lambda (gt? 1 0)) $0) (range (#(lambda (fold $0 1 (lambda (lambda (* $1 $0))))) (map (lambda $0) $0))) (lambda (lambda (range #(+ 1 1)))))))")
        t = p.infer()
        print(t)
        print(LearnedFeatureExtractor(tasks).taskOfProgram(p, t))
        assert False

    explorationCompression(baseGrammar, train, testingTasks=test, **args)
