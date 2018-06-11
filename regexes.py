# analog of list.py for regex tasks. Responsible for actually running the task.

from ec import explorationCompression, commandlineArguments, Task
from grammar import Grammar
#from utilities import eprint, testTrainSplit, numberOfCPUs, flatten
from utilities import eprint, numberOfCPUs, flatten, fst, testTrainSplit, POSITIVEINFINITY
from makeRegexTasks import makeOldTasks, makeLongTasks, makeShortTasks, makeWordTasks, makeNumberTasks
from regexPrimitives import basePrimitives, altPrimitives
#from program import *
from recognition import HandCodedFeatureExtractor, MLPFeatureExtractor, RecurrentFeatureExtractor, JSONFeatureExtractor
import random
from type import tpregex
import math



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


class MyJSONFeatureExtractor(JSONFeatureExtractor):
    N_EXAMPLES = 5

    def _featuresOfProgram(self, program, tp):
        try:
            preg = program.evaluate([])
            # if 'left_paren' in program.show(False):
            #eprint("string_pregex:", string_pregex)
            #eprint("string_pregex:", string_pregex)

        except IndexError:
            # free variable
            return None
        except Exception as e:
            eprint("Exception during evaluation:", e)
            if "Attempt to evaluate fragment variable" in e:
                eprint("program (bc fragment error)", program)
            return None

        examples = []

        for _ in range(self.N_EXAMPLES * 5):  # oh this is arbitrary ig

            try:
                y = preg.sample()  # TODO

                #this line should keep inputs short, so that helmholtzbatch can be large
                #allows it to try other samples
                #(Could also return None off the bat... idk which is better)
                #if len(y) > 20:
                #    continue
                #eprint(tp, program, x, y)
                examples.append(y)
            except BaseException:
                continues
            if len(examples) >= self.N_EXAMPLES:
                break
        else:
            return None
        return examples  # changed to list_features(examples) from examples


def regex_options(parser):
    parser.add_argument("--maxTasks", type=int,
                        default=500,
                        help="truncate tasks to fit within this boundary")
    parser.add_argument(
        "--maxExamples",
        type=int,
        default=10,
        help="truncate number of examples per task to fit within this boundary")
    parser.add_argument("--tasks",
                        default="old",
                        help="which tasks to use",
                        choices=["old", "short", "long", "words","number"])
    parser.add_argument("--primitives",
                        default="base",
                        help="Which primitive set to use",
                        choices=["base", "alt1"])
    parser.add_argument("--extractor", type=str,
                        choices=["hand", "deep", "learned", "json"],
                        default="json")  # if i switch to json it breaks
    parser.add_argument("--split", metavar="TRAIN_RATIO",
                        type=float,
                        default=0.8,
                        help="split test/train")
    parser.add_argument("-H", "--hidden", type=int,
                        default=16,
                        help="number of hidden units")
    parser.add_argument("--likelihoodModel",
                        default="probabilistic",
                        help="likelihood Model",
                        choices=["probabilistic", "all-or-nothing"])
    parser.add_argument("--topk_use_map",
                        dest="topk_use_map",
                        action="store_true")
    parser.add_argument("--debug",
                        dest="debug",
                        action="store_true"
                        )
    """parser.add_argument("--stardecay",
                        type=float,
                        dest="stardecay",
                        default=0.5,
                        help="p value for kleenestar and plus")"""

# Lucas recommends putting a struct with the definitions of the primitives here.
# TODO:
# Build likelihood funciton
# modify NN
# make primitives
# make tasks


if __name__ == "__main__":
    args = commandlineArguments(
        frontierSize=None, activation='sigmoid', iterations=10,
        a=3, maximumFrontier=10, topK=2, pseudoCounts=10.0, #try 1 0.1 would make prior uniform
        helmholtzRatio=0.5, structurePenalty=1., #try 
        CPUs=numberOfCPUs(),
        extras=regex_options)

    regexTasks = {"old": makeOldTasks,
                "short": makeShortTasks,
                "long": makeLongTasks,
                "words": makeWordTasks,
                "number": makeNumberTasks
                }[args.pop("tasks")]

    tasks = regexTasks()  # TODO
    eprint("Generated", len(tasks), "tasks")

    maxTasks = args.pop("maxTasks")
    if len(tasks) > maxTasks:
        eprint("Unwilling to handle {} tasks, truncating..".format(len(tasks)))
        random.seed(42)
        random.shuffle(tasks)
        del tasks[maxTasks:]

    maxExamples = args.pop("maxExamples")
    for task in tasks:
        if len(task.examples) > maxExamples:
            task.examples = task.examples[:maxExamples]

    split = args.pop("split")
    test, train = testTrainSplit(tasks, split)
    eprint("Split tasks into %d/%d test/train" % (len(test), len(train)))

    # from list stuff
    primtype = args.pop("primitives")
    prims = {"base": basePrimitives,
             "alt1": altPrimitives}[primtype]

    extractor = {
        "hand": HandCodedFeatureExtractor,
        "learned": LearnedFeatureExtractor,
        "json": MyJSONFeatureExtractor
    }[args.pop("extractor")]

    extractor.H = args.pop("hidden")
    extractor.USE_CUDA = args["cuda"]


    from time import gmtime, strftime
    timestr = strftime("%m%d%H%M%S", gmtime())

    #stardecay = args.stardecay
    #stardecay = args.pop('stardecay')
    #decaystr = 'd' + str(stardecay)

    args.update({
        "featureExtractor": extractor,
        "outputPrefix": "experimentOutputs/regex" + primtype + timestr, #+ decaystr,
        "evaluationTimeout": 1.0,  # 0.005,
        "topk_use_map": False,
        "maximumFrontier": 10,
        "solver": "python",
        "compressor": "rust"
    })
    ####


        # use the
    #prim_list = prims(stardecay)
    prim_list = prims()
    n_base_prim = len(prim_list) - 5.
    specials = ["r_kleene", "r_plus", "r_maybe", "r_alt", "r_concat"]

    productions = [
        (math.log(0.5 / n_base_prim),
         prim) if prim.name not in specials else (
            math.log(0.10),
            prim) for prim in prim_list]


    baseGrammar = Grammar.fromProductions(productions)
    #baseGrammar = Grammar.uniform(prims())

    #for i in range(100):
    #    eprint(baseGrammar.sample(tpregex))

    #eprint(baseGrammar)
    #explore
    test_stuff = args.pop("debug")
    if test_stuff:
        eprint(baseGrammar)
        eprint("sampled programs from prior:")
        for i in range(100): #100
            eprint(baseGrammar.sample(test[0].request,maximumDepth=1000))
        eprint("""half the probability mass is on higher-order primitives.
Therefore half of enumerated programs should have more than one node.
However, we do not observe this.
Instead we see a very small fraction of programs have more than one node. 
So something seems to be wrong with grammar.sample.

Furthermore: observe the large print statement above. 
This prints the candidates for sampleDistribution in grammar.sample.
the first element of each tuple is the probability passed into sampleDistribution.
Half of the probability mass should be on the functions, but instead they are equally 
weighted with the constants. If you look at the grammar above, this is an error!!!!
""")
        assert False

    explorationCompression(baseGrammar, train,
                            testingTasks = test,
                            **args)
