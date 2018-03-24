from ec import explorationCompression, commandlineArguments, Task
from grammar import Grammar
from utilities import eprint, testTrainSplit, numberOfCPUs
from makeTextTasks import makeTasks, delimiters
from textPrimitives import primitives
from program import *
from recognition import *

import random

class LearnedFeatureExtractor(RecurrentFeatureExtractor):
    def __init__(self, tasks):
        def serialize(x):
            if isinstance(x,str): return [x]
            assert isinstance(x,list)
            if isinstance(x[0],str): return x
            assert isinstance(x[0],list)
            serialization = ["LIST"]
            for s in x:
                serialization.append("LISTDELIMITER")
                serialization += serialize(s)
            return serialization
            
        def tokenize(examples, _):
            return [ ((serialize(x),), serialize(y))
                     for (x,),y in examples]

        lexicon = {c
                   for t in tasks
                   for (x,),y in tokenize(t.examples, None)
                   for c in x + y }
                
        super(LearnedFeatureExtractor, self).__init__(lexicon = list(lexicon),
                                                      H = 64,
                                                      tasks = tasks,
                                                      bidirectional = True,
                                                      tokenize = tokenize)


if __name__ == "__main__":
    tasks = makeTasks()
    eprint("Generated",len(tasks),"tasks")

    test, train = testTrainSplit(tasks, 0.2)
    eprint("Split tasks into %d/%d test/train"%(len(test),len(train)))

    baseGrammar = Grammar.uniform(primitives)
    
    explorationCompression(baseGrammar, train,
                           testingTasks = test,
                           outputPrefix = "experimentOutputs/text",
                           evaluationTimeout = 0.0005,
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
