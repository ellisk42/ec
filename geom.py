from ec import explorationCompression, commandlineArguments
from grammar import Grammar
from utilities import eprint, testTrainSplit, numberOfCPUs
from makeGeomTasks import makeTasks
from geomPrimitives import primitives, tcanvas

import torch
import subprocess
import torch.nn as nn

from recognition import variable


# : Task -> feature list
class GeomFeatureCNN(nn.Module):
    def __init__(self, tasks, cuda=False, H=10):
        super(GeomFeatureCNN, self).__init__()

        self.net1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.ReLU()
        )
        self.net2 = nn.Sequential(
            nn.Linear(256, 120),
            nn.Linear(120, H)
        )

        self.outputDimensionality = H
        self.out = nn.Sequential(self.net1)

    def featuresOfTask(self, t):  # Take a task and returns [features]
        x, y = 16, 16  # Should not hardocode these.
        onlyTask = t.examples[0][1]
        floatOnlyTask = map(float, onlyTask)
        reshaped = [floatOnlyTask[i:i+x]
                    for i in range(0, len(floatOnlyTask), y)]
        variabled = variable(reshaped).float()
        variabled = torch.unsqueeze(variabled, 0)
        variabled = torch.unsqueeze(variabled, 0)
        output = self.net1(variabled)
        output = output.view(256)
        output = self.net2(output)
        return output

    def featuresOfProgram(self, p, t):  # Won't fix for geom
        shape = ""
        if t == tcanvas:
            try:
                shape = subprocess.check_output(['./geomDrawLambdaString',
                                                 str(p)])
            except OSError as exc:
                raise exc
        else:
            assert(False)
# Now I need to do something, anything with shape...
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
                           evaluationTimeout=0.1,
                           **commandlineArguments(
                               steps=10,
                               iterations=2,
                               useRecognitionModel=True,
                               helmholtzRatio=1.,
                               featureExtractor=GeomFeatureCNN,
                               topK=2,
                               maximumFrontier=10,
                               CPUs=numberOfCPUs(),
                               pseudoCounts=10.0))
