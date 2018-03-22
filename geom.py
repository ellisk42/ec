from ec import explorationCompression, commandlineArguments
from grammar import Grammar
from utilities import eprint, testTrainSplit, numberOfCPUs
from makeGeomTasks import makeTasks
from geomPrimitives import primitives, tcanvas
from math import log

import torch
import png
import time
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
            nn.Linear(2304, 120),
            nn.Linear(120, 84),
            nn.Linear(84, H)
        )

        self.mean = []

        self.outputDimensionality = H

    def forward(self, v):
        x, y = 32, 32  # Should not hardocode these.
        floatOnlyTask = map(float, v)
        reshaped = [floatOnlyTask[i:i+x]
                    for i in range(0, len(floatOnlyTask), y)]
        variabled = variable(reshaped).float()
        variabled = torch.unsqueeze(variabled, 0)
        variabled = torch.unsqueeze(variabled, 0)
        output = self.net1(variabled)
        output = output.view(-1, 2304)
        output = self.net2(output)
        return output.clamp(min=0)

    def featuresOfTask(self, t):  # Take a task and returns [features]
        return self(t.examples[0][1])

    def featuresOfProgram(self, p, t):  # Won't fix for geom
        if t == tcanvas:
            try:
                output = subprocess.check_output(['./geomDrawLambdaString',
                                                 p.evaluate([])]).split("\n")
                shape = output[0]
                bigShape = map(float, output[1])
            except OSError as exc:
                raise exc
        else:
            assert(False)
        try:
            self.mean += [bigShape]
            return self(shape)
        except ValueError:
            return None

    def finish(self):
        mean = [log(float(sum(col))/len(col)) for col in zip(*self.mean)]
        mi = min(mean)
        ma = max(mean)
        mean = [(x - mi + (1/255)) / (ma - mi) for x in mean]
        img = [(int(x*254), int(x*254), int(x*254)) for x in mean]
        img = [img[i:i+64] for i in range(0, 64*64, 64)]
        img = [tuple([e for t in x for e in t]) for x in img]
        fname = 'dream-'+(str(int(time.time())))+'.png'
        f = open(fname, 'wb')
        w = png.Writer(64, 64)
        w.write(f, img)
        f.close()
        self.mean = []


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
                               steps=500,
                               iterations=10,
                               useRecognitionModel=True,
                               helmholtzRatio=0.0,
                               featureExtractor=GeomFeatureCNN,
                               topK=2,
                               maximumFrontier=50,
                               CPUs=numberOfCPUs(),
                               pseudoCounts=10.0))
