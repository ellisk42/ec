import cPickle as pickle
import random
from ec import explorationCompression, commandlineArguments
from utilities import eprint, numberOfCPUs
from grammar import Grammar
from task import RegressionTask
from type import *
from listPrimitives import primitives
from makeListTasks import list_features, N_EXAMPLES

def retrieveTasks():
    with open("data/list_tasks.pkl") as f:
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
        for _ in xrange(2000):
            x = sample()
            try:
                y = e(x)
                examples.append(((x,), y))
            except: continue
            if len(examples) >= N_EXAMPLES: break
        else: return None
        return RegressionTask.standardizeFeatures(averages, deviations, list_features(examples))
    return featureExtractor


if __name__ == "__main__":
    tasks = retrieveTasks()
    eprint("Got {} list tasks".format(len(tasks)))

    statistics = RegressionTask.standardizeTasks(tasks)
    featureExtractor = makeFeatureExtractor(statistics)

    baseGrammar = Grammar.uniform(primitives)
    explorationCompression(baseGrammar, tasks,
                           outputPrefix="experimentOutputs/list",
                           **commandlineArguments(frontierSize=15000,
                                                  activation='sigmoid',
                                                  a=2,
                                                  maximumFrontier=2,
                                                  topK=2,
                                                  CPUs=numberOfCPUs(),
                                                  featureExtractor=featureExtractor,
                                                  iterations=10,
                                                  pseudoCounts=10.0))
