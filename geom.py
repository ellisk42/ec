from ec import explorationCompression, commandlineArguments
from grammar import Grammar
from utilities import eprint, testTrainSplit, numberOfCPUs
from makeGeomTasks import makeTasks
from geomPrimitives import primitives
from math import log

import torch
import png
import time
import subprocess
import os
import torch.nn as nn

from recognition import variable


# : Task -> feature list
class GeomFeatureCNN(nn.Module):
    def __init__(self, tasks, cuda=False, H=10):
        super(GeomFeatureCNN, self).__init__()

        self.net1 = nn.Sequential( # Maybe dont do that but rather rescale first
            nn.Conv2d(1, 6, kernel_size=(10, 10), stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

        self.net2 = nn.Sequential(
            nn.Linear(16*5*5, 120),  # Hardocde the first one I guess :/
            nn.Linear(120, 84),
            nn.Linear(84, H)
        )

        self.mean = [0]*(256*256)
        self.count = 0
        self.sub = "./dreams/" + str(int(time.time()))

        self.outputDimensionality = H

    def forward(self, v):
        x, y = 64, 64  # Should not hardocode these.
        floatOnlyTask = map(float, v)
        reshaped = [floatOnlyTask[i:i+x]
                    for i in range(0, len(floatOnlyTask), y)]
        variabled = variable(reshaped).float()
        variabled = torch.unsqueeze(variabled, 0)
        variabled = torch.unsqueeze(variabled, 0)
        output = self.net1(variabled)
        output = output.view(-1, 16*5*5)
        output = self.net2(output).clamp(min=0)
        output = torch.squeeze(output)
        return output

    def featuresOfTask(self, t):  # Take a task and returns [features]
        return self(t.examples[0][1])

    def featuresOfProgram(self, p, t):  # Won't fix for geom
        if not os.path.exists(self.sub):
                os.makedirs(self.sub)
        try:
            output = subprocess.check_output(['./geomDrawLambdaString',
                                             self.sub +
                                              "/" +
                                              str(self.count) +
                                              ".png",
                                             p.evaluate([])]).split("\n")
            shape = map(float, output[0].split(','))
            bigShape = map(float, output[1].split(','))
        except OSError as exc:
            raise exc
        try:
            self.mean = [x+y for x, y in zip(self.mean, bigShape)]
            self.count += 1
            return self(shape)
        except ValueError:
            return None

    def finish(self):
        if self.count > 0:
            mean = [log(1+float(x/self.count)) for x in self.mean]
            mi = min(mean)
            ma = max(mean)
            mean = [(x - mi + (1/255)) / (ma - mi) for x in mean]
            img = [(int(x*254), int(x*254), int(x*254)) for x in mean]
            img = [img[i:i+256] for i in range(0, 256*256, 256)]
            img = [tuple([e for t in x for e in t]) for x in img]
            fname = self.sub+"/"+str(self.count)+"_sum.png"
            f = open(fname, 'wb')
            w = png.Writer(256, 256)
            w.write(f, img)
            f.close()
            self.mean = [0]*(256*256)
            self.count = 0


if __name__ == "__main__":
    tasks = makeTasks()
    eprint("Generated", len(tasks), "tasks")

    test, train = testTrainSplit(tasks, 0.5)
    eprint("Split tasks into %d/%d test/train" % (len(test), len(train)))

    baseGrammar = Grammar.uniform(primitives)

    explorationCompression(baseGrammar, train,
                           testingTasks=test,
                           outputPrefix="experimentOutputs/geom",
                           compressor="rust",
                           evaluationTimeout=0.01,
                           **commandlineArguments(
                               steps=200,
                               a=1,
                               iterations=10,
                               useRecognitionModel=True,
                               helmholtzRatio=0.5,
                               helmholtzBatch=500,
                               featureExtractor=GeomFeatureCNN,
                               topK=2,
                               maximumFrontier=500,
                               CPUs=numberOfCPUs(),
                               pseudoCounts=10.0))
