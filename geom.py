from ec import explorationCompression, commandlineArguments
from grammar import Grammar
from utilities import eprint, testTrainSplit, numberOfCPUs
from makeGeomTasks import makeTasks
from geomPrimitives import primitives

import torch
import torch.nn as nn

from recognition import variable


# : Task -> feature list
class GeomFeatureCNN(nn.Module):
    def __init__(self, tasks, cuda=False, H=4):
        super(GeomFeatureCNN, self).__init__()

        size = 16  # Compute this maybe?

        self.conv = nn.Conv2d(size, H, 1)
        self.relu = nn.ReLU()
        self.out = nn.Sequential(self.conv, self.relu)

        self.outputDimensionality = size*H
        self.hidden = self.out

    def featuresOfTask(self, t):  # Take a task and returns [features]
        x, y = 16, 16  # Should not hardocode these.
        onlyTask = t.examples[0][1]
        floatOnlyTask = map(float, onlyTask)
        reshaped = [floatOnlyTask[i:i+x]
                    for i in range(0, len(floatOnlyTask), y)]
        variabled = variable(reshaped).float()
        variabled = torch.unsqueeze(variabled, -1)
        variabled = torch.unsqueeze(variabled, 0)
        x = self.out(variabled)
        x = torch.squeeze(x).view(4*16).clamp(min=0)
        return x

    def featuresOfProgram(self, p, t):  # Won't fix for geom
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
                               steps=100000,
                               iterations=10,
                               useRecognitionModel=True,
                               helmholtzRatio=0.0,
                               featureExtractor=GeomFeatureCNN,
                               topK=2,
                               maximumFrontier=100,
                               CPUs=numberOfCPUs(),
                               pseudoCounts=10.0))
