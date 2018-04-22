from ec import explorationCompression, commandlineArguments, Task
from grammar import Grammar
from utilities import eprint, testTrainSplit, numberOfCPUs
from makeTextTasks import makeTasks, delimiters
from textPrimitives import primitives
from listPrimitives import bootstrapTarget
from program import *
from recognition import *

import random

class LearnedFeatureExtractor(RecurrentFeatureExtractor):
    def tokenize(self, examples):
        return examples

    def __init__(self, tasks):
        lexicon = {c
                   for t in tasks
                   for xs,y in self.tokenize(t.examples)
                   for c in reduce(lambda u,v: u+v, list(xs) + [y]) }
                
        super(LearnedFeatureExtractor, self).__init__(lexicon = list(lexicon),
                                                      H = 64,
                                                      tasks = tasks,
                                                      bidirectional = True)


if __name__ == "__main__":
    tasks = makeTasks()
    eprint("Generated",len(tasks),"tasks")

    test, train = testTrainSplit(tasks, 0.9)
    eprint("Split tasks into %d/%d test/train"%(len(test),len(train)))

    baseGrammar = Grammar.uniform(primitives + bootstrapTarget())
    
    explorationCompression(baseGrammar, train,
                           testingTasks = test,
                           outputPrefix = "experimentOutputs/text",
                           evaluationTimeout = 0.0005,
                           compressor="pypy", # 
                           **commandlineArguments(
                               steps = 500,
                               iterations = 10,
                               helmholtzRatio = 0.5,
                               topK = 2,
                               maximumFrontier = 2,
                               structurePenalty = 10.,
                               a = 3,
                               activation = "relu",
                               CPUs = numberOfCPUs(),
                               featureExtractor = LearnedFeatureExtractor,
                               pseudoCounts = 10.0))
