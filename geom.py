from ec import explorationCompression, commandlineArguments
from grammar import Grammar
from utilities import eprint, testTrainSplit, numberOfCPUs
from makeGeomTasks import makeTasks
from geomPrimitives import primitives


if __name__ == "__main__":
    tasks = makeTasks()
    eprint("Generated", len(tasks), "tasks")

    test, train = testTrainSplit(tasks, 0.5)
    eprint("Split tasks into %d/%d test/train" % (len(test), len(train)))

    baseGrammar = Grammar.uniform(primitives)

    explorationCompression(baseGrammar, train,
                           testingTasks=test,
                           outputPrefix="experimentOutputs/geom",
                           evaluationTimeout=None,
                           **commandlineArguments(
                               steps=5,
                               useRecognitionModel=False,
                               iterations=5,
                               helmholtzRatio=0.5,
                               topK=2,
                               maximumFrontier=500,
                               structurePenalty=10.,
                               a=3,
                               activation="relu",
                               CPUs=numberOfCPUs(),
                               featureExtractor=None,
                               pseudoCounts=10.0))
