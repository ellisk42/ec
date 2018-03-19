from ec import explorationCompression, commandlineArguments
from grammar import Grammar
from utilities import eprint, testTrainSplit, numberOfCPUs
from makeGeomTasks import makeTasks
from geomPrimitives import primitives

import torch.nn as nn

from recognition import variable


# : Task -> feature list
class GeomFeatureCNN(nn.Module):
    def __init__(self, tasks, cuda=False, H=16):
        super(GeomFeatureCNN, self).__init__()

        size = 16  # Compute this maybe?

        self.conv = nn.Conv1d(size, H, H)
        self.relu = nn.ReLU()
        self.out = nn.Sequential(self.conv, self.relu)

        self.outputDimensionality = H
        self.hidden = self.conv

    def featuresOfTask(self, t):  # Take a task and returns [features]
        x = 16  # Should not hardocode these.
        y = 16
        onlyTask = t.examples[0][1]
        floatOnlyTask = map(float, onlyTask)
        reshaped = [[floatOnlyTask[i:i+x]
                    for i in range(0, len(floatOnlyTask), y)]]
        variabled = variable(reshaped).float()
        x = self.out(variabled)
        x = x.view(16)
        return x.clamp(min=0)

    def featuresOfProgram(self, p, t):  # Won't do for geom
        return None


if __name__ == "__main__":
    tasks = makeTasks()
    eprint("Generated", len(tasks), "tasks")

    test, train = testTrainSplit(tasks, 0.5)
    eprint("Split tasks into %d/%d test/train" % (len(test), len(train)))

    baseGrammar = Grammar.uniform(primitives)

    explorationCompression(baseGrammar, train,
                           testingTasks=test,
                           outputPrefix="experimentOutputs/geom",
                           compressor="pypy",
                           evaluationTimeout=0.01,
                           **commandlineArguments(
                               steps=5,
                               iterations=5,
                               useRecognitionModel=True,
                               helmholtzRatio=0.0,
                               featureExtractor=GeomFeatureCNN,
                               topK=2,
                               maximumFrontier=200,
                               CPUs=numberOfCPUs(),
                               pseudoCounts=10.0))
