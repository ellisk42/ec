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
        hidden = nn.Linear(16*16, H) # TODO Do not hardcode 16
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
        # features = self._featuresOfProgram(p,t)
        # if features is None: return None
        # f = variable([ (f - self.averages[j])/self.deviations[j] for j,f in enumerate(features) ], cuda=self.use_cuda).float()
        # return self.hidden(f).clamp(min = 0)

if __name__ == "__main__":
    tasks = makeTasks()
    eprint("Generated", len(tasks), "tasks")

    test, train = testTrainSplit(tasks, 0.5)
    eprint("Split tasks into %d/%d test/train" % (len(test), len(train)))

    baseGrammar = Grammar.uniform(primitives)

    explorationCompression(baseGrammar, train,
                           testingTasks=test,
                           outputPrefix="experimentOutputs/geom",
                           evaluationTimeout=0.01,
                           **commandlineArguments(
                               steps=5,
                               iterations=5,
                               useRecognitionModel=True,
                               helmholtzRatio=0.,
                               featureExtractor=GeomFeatureCNN,
                               topK=2,
                               maximumFrontier=250,
                               CPUs=numberOfCPUs(),
                               pseudoCounts=10.0))
