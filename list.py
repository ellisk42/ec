import cPickle as pickle
import random
from ec import explorationCompression, commandlineArguments
from utilities import eprint, numberOfCPUs
from grammar import Grammar
from task import RegressionTask
from type import *
from listPrimitives import primitives
from makeListTasks import list_features, N_EXAMPLES

def retrieveTasks(filename):
    with open(filename) as f:
        return pickle.load(f)


def makeFeatureExtractor((averages, deviations)):
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

    def featureExtractor(program, tp):
        e = program.evaluate([])
        examples = []
        if isListFunction(tp):
            sample = lambda: random.sample(xrange(30), random.randint(0, 8))
        elif isIntFunction(tp):
            sample = lambda: random.randint(0, 20)
        else: return None
        for _ in xrange(N_EXAMPLES*5):
            x = sample()
            try:
                y = e(x)
                examples.append(((x,), y))
            except: continue
            if len(examples) >= N_EXAMPLES: break
        else: return None
        return RegressionTask.standardizeFeatures(averages, deviations, list_features(examples))
    return featureExtractor


def list_clis(parser):
    parser.add_argument("--dataset", type=str,
        default="data/list_tasks.pkl",
        help="location of pickled list function dataset")
    parser.add_argument("--maxTasks", type=int,
        default=1000,
        help="truncate tasks to fit within this boundary")

if __name__ == "__main__":
    args = commandlineArguments(
        frontierSize=10000, activation='sigmoid', iterations=10,
        a=3, topK=3, pseudoCounts=10.0,
        CPUs=numberOfCPUs(),
        extras=list_clis)

    maxTasks = args["maxTasks"]
    tasks = retrieveTasks(args["dataset"])
    eprint("Got {} list tasks".format(len(tasks)))
    if len(tasks) > maxTasks:
        eprint("Unwilling to handle more than {} tasks, truncating..".format(maxTasks))
        random.seed(42)
        random.shuffle(tasks)
        del tasks[maxTasks:]

    del args["dataset"]
    del args["maxTasks"]

    statistics = RegressionTask.standardizeTasks(tasks)
    args["featureExtractor"] = makeFeatureExtractor(statistics)

    baseGrammar = Grammar.uniform(primitives)
    explorationCompression(baseGrammar, tasks, outputPrefix="experimentOutputs/list", **args)
