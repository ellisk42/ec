import random
from collections import defaultdict
import json
import math
import os
import datetime
import torch
import numpy as np
from torch import nn

from dreamcoder.dreamcoder import explorationCompression
from dreamcoder.utilities import eprint, flatten, testTrainSplit
from dreamcoder.grammar import Grammar
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
from dreamcoder.domains.list.listPrimitives import basePrimitives, primitives, McCarthyPrimitives, bootstrapTarget_extra, no_length
from dreamcoder.recognition import RecurrentFeatureExtractor
from dreamcoder.domains.list.makeListTasks import make_list_bootstrap_tasks, sortBootstrap, EASYLISTTASKS

from dreamcoder.domains.misc.deepcoderPrimitives import deepcoderPrimitives
from dreamcoder.domains.list.makeDeepcoderData import DeepcoderTaskloader


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


class ListFeatureExtractor(RecurrentFeatureExtractor):
    special = None
    def __init__(self, maximumLength, cuda=True, H=64):
        #T=3
        # self.deepcoder_taskloader = DeepcoderTaskloader(
        #    f'dreamcoder/domains/list/DeepCoder_data/T{3}_A2_V512_L10_train_perm.txt',
        #    allowed_requests=[arrow(tlist(tint),tlist(tint))],
        #    repeat=True,
        #    micro=None,
        #    )
        self.lexicon = {"LIST_START", "LIST_END", "?"}.union(set(range(-64, 64)))
        # add all non-arrow primitives ie things that should count as concrete values as opposed to arrows that are represented with NMs
        self.lexicon = self.lexicon.union({p.value for p in deepcoderPrimitives() if not p.tp.isArrow()})

        self.maximumLength = maximumLength

        # encodes a [int]
        self.list_encoder = nn.GRU(input_size=H, hidden_size=H, num_layers=1, bidirectional=True)

        # encodes a task (set of inputs and outputs)
        self.task_encoder = nn.GRU(input_size=H, hidden_size=H, num_layers=1, bidirectional=True)

        #self.recomputeTasks = True

        super().__init__(
            lexicon=list(self.lexicon),
            tasks=[],
            cuda=cuda,
            H=H,
            bidirectional=True)

        self.lexicon = set(self.lexicon)

    def tokenize(self, examples):
        def sanitize(l): 
            ret = []
            for z_ in l:
                for z in (z_ if isinstance(z_, list) else [z_]):
                    if (z in self.lexicon) or isinstance(z, torch.Tensor):
                        ret.append(z)
                    else:
                        #print (f"WARNING: the following token was not tokenized: {z}")
                        ret.append("?")
            return ret

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
    def inputFeatures(self,task):
        tokenized = self.tokenize(task.examples)
        assert tokenized

        # this deletes any LIST_START LIST_END inserted by .tokenize()
        tokenized = [(ex[0],[]) for ex in tokenized]

        e = self.examplesEncoding(tokenized)
        return e
    def outputFeatures(self,task):
        tokenized = self.tokenize(task.examples)
        assert tokenized

        # this deletes any LIST_START LIST_END inserted by .tokenize()
        tokenized = [([],ex[1]) for ex in tokenized]

        e = self.examplesEncoding(tokenized)
        return e
    
    def encodeValue(self, val):
        """
        val :: a concrete value that shows up as an intermediate value
            during computation and converts it to a vector. Note that this is
            always a list of N such values sine we're evaluating on all N examples
            at once. val :: [int] or [bool] or [[int]] where the outer list
            is always the list of examples.
        returns :: Tensor[num_exs,H]
        """
        is_list = lambda v: isinstance(v,(list,tuple))
        is_int = lambda v: isinstance(v,(int,np.integer))
        is_bool = lambda v: isinstance(v,(bool,np.bool_))

        assert is_list(val)
        #assert len(val) == self.deepcoder_taskloader.N
        if is_list(val[0]):
            assert len(val[0]) == 0 or is_int(val[0][0]) # make sure its [[int]]
            class FakeTask: pass
            fake_task = FakeTask()
            fake_task.examples = [((v,),[]) for v in val] # :: [(([Int],),[])] in the single argument case
            vec = self.inputFeatures(fake_task) # [5,H]
            return vec
        
        # val is one concrete non-list value per example
        for i,v in enumerate(val):
            if v not in self.symbolToIndex:
                print("converthing {v} to <UNK>")
                v[i] = "?"
        res = self.encoder([self.symbolToIndex[v] for v in val],cuda=self.use_cuda)
        return res

    def sampleHelmholtzTask(self, request, motifs=[]):
        assert False # disabling for now
        # NOTE: we ignore the `request` argument
        if motifs != []:
            raise NotImplementedError
        try:
            program, task = self.deepcoder_taskloader.getTask()
        except StopIteration:
            return None,None
        return program, task



def train_necessary(t):
    if t.name in {"head", "is-primes", "len", "pop", "repeat-many", "tail", "keep primes", "keep squares"}:
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
        "--noMap", action="store_true", default=False,
        help="Disable built-in map primitive")
    parser.add_argument(
        "--noUnfold", action="store_true", default=False,
        help="Disable built-in unfold primitive")
    parser.add_argument(
        "--noLength", action="store_true", default=False,
        help="Disable built-in length primitive")
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
                        choices=["McCarthy", "base", "rich", "common", "noLength", "deepcoder"])
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


def main(args):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on manipulating sequences of numbers.
    """
    random.seed(args.pop("random_seed"))

    dataset = args.pop("dataset")
    tasks = {
        "Lucas-old": lambda: retrieveJSONTasks("data/list_tasks.json") + sortBootstrap(),
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

    def isIdentityTask(t):
        return all( len(xs) == 1 and xs[0] == y for xs, y in t.examples  )
    eprint("Removed", sum(isIdentityTask(t) for t in tasks), "tasks that were just the identity function")
    tasks = [t for t in tasks if not isIdentityTask(t) ]

    prims = {"base": basePrimitives,
             "McCarthy": McCarthyPrimitives,
             "common": bootstrapTarget_extra,
             "noLength": no_length,
             "deepcoder": deepcoderPrimitives,
             "rich": primitives}[args.pop("primitives")]()
    haveLength = not args.pop("noLength")
    haveMap = not args.pop("noMap")
    haveUnfold = not args.pop("noUnfold")
    eprint(f"Including map as a primitive? {haveMap}")
    eprint(f"Including length as a primitive? {haveLength}")
    eprint(f"Including unfold as a primitive? {haveUnfold}")
    baseGrammar = Grammar.uniform([p
                                   for p in prims
                                   if (p.name != "map" or haveMap) and \
                                   (p.name != "unfold" or haveUnfold) and \
                                   (p.name != "length" or haveLength)])

    extractor = {
        "learned": LearnedFeatureExtractor,
    }[args.pop("extractor")]
    extractor.H = args.pop("hidden")

    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/list/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)
    
    args.update({
        "featureExtractor": extractor,
        "outputPrefix": "%s/list"%outputDirectory,
        "evaluationTimeout": 0.0005,
    })
    

    eprint("Got {} list tasks".format(len(tasks)))
    split = args.pop("split")
    if split:
        train_some = defaultdict(list)
        for t in tasks:
            necessary = train_necessary(t)
            if not necessary:
                continue
            if necessary == "some":
                train_some[t.name.split()[0]].append(t)
            else:
                t.mustTrain = True
        for k in sorted(train_some):
            ts = train_some[k]
            random.shuffle(ts)
            ts.pop().mustTrain = True

        test, train = testTrainSplit(tasks, split)
        if True:
            test = [t for t in test
                    if t.name not in EASYLISTTASKS]

        eprint(
            "Alotted {} tasks for training and {} for testing".format(
                len(train), len(test)))
    else:
        train = tasks
        test = []
    
    T=3
    testing_taskloader = DeepcoderTaskloader(
        #'dreamcoder/domains/list/DeepCoder_data/T1_A2_V512_L10_test_perm.txt', #TODO <- careful this is set to `train` instead of `test` for an ultra simple baseline
        #'dreamcoder/domains/list/DeepCoder_data/T1_A2_V512_L10_test_perm.txt',
        #'dreamcoder/domains/list/DeepCoder_data/T2_A2_V512_L10_test_perm.txt',
        f'dreamcoder/domains/list/DeepCoder_data/T{T}_A2_V512_L10_test_perm.txt',
        allowed_requests=[arrow(tlist(tint),tlist(tint))],
        repeat=True,
        micro=None,
        )

    test = [testing_taskloader.getTask()[1] for _ in range(8)]

    explorationCompression(baseGrammar,[], testingTasks=test, **args)
