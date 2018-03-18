from ec import explorationCompression, commandlineArguments
from grammar import Grammar
from utilities import eprint, testTrainSplit, numberOfCPUs
from makeGeomTasks import makeTasks
from geomPrimitives import primitives

import torch
import torch.nn as nn
import torch.optim as optimization
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from recognition import variable

# : Task -> feature list
class GeomFeatureCNN(nn.Module):
    def __init__(self, tasks, cuda=False, H=16):
        super(GeomFeatureCNN, self).__init__()

        self.tasks = tasks
        self.use_cuda = cuda

        self.outputDimensionality = H
        hidden = nn.Linear(len(tasks[0].examples[0][1]), H)
        if cuda:
            hidden = hidden.cuda()
        else:
            hidden = hidden.float()
        self.hidden = hidden

    def featuresOfTask(self, t): # Take a task and returns [features]
        onlyTask = t.examples[0][1]
        floatOnlyTask = map(float,onlyTask)
        variabled = variable(floatOnlyTask).float()
        return self.hidden(variabled).clamp(min = 0)

    def featuresOfProgram(self, p, t):
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
                               iterations=2,
                               useRecognitionModel=True,
                               helmholtzRatio=0.5,
                               featureExtractor=GeomFeatureCNN,
                               topK=2,
                               maximumFrontier=5,
                               CPUs=numberOfCPUs(),
                               pseudoCounts=10.0))
