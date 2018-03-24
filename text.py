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
        lexicon = set([ c
                        for t in tasks
                        for (x,),y in t.examples
                        for xp in [x if isinstance(x,str) else "".join(x)]
                        for yp in [y if isinstance(y,str) else "".join(y)] 
                        for c in xp+yp ] + ["LIST","LISTDELIMITER"])
        def serialize(x):
            if isinstance(x,str): return x
            assert isinstance(x,list)
            serialization = ["LIST"]
            for s in x:
                serialization.append("LISTDELIMITER")
                serialization += [c for c in s ]
            return serialization
            
        def tokenize(examples, _):
            return [ ((serialize(x),), serialize(y))
                     for (x,),y in examples]            
                
        super(LearnedFeatureExtractor, self).__init__(lexicon = list(lexicon),
                                                      H = 16,
                                                      tasks = tasks,
                                                      bidirectional = True,
                                                      tokenize = tokenize)


if __name__ == "__main__":
    tasks = makeTasks()
    eprint("Generated",len(tasks),"tasks")

    test, train = testTrainSplit(tasks, 0.02)
    eprint("Split tasks into %d/%d test/train"%(len(test),len(train)))

    # target = "Apply double delimited by '<' to input delimited by '>'"
    # program = Program.parse("(lambda (join (chr->str '<') (map (lambda (++ $0 $0)) (split '<' $0))))")
    # eprint(program)
    # tasks = {t.name: t for t in tasks }
    baseGrammar = Grammar.uniform(primitives)
    # eprint(baseGrammar.logLikelihood(tasks[target].request, program))
    # callCompiled(enumerateForTask,
    #              baseGrammar, tasks[target],
    #              maximumFrontier = 2,
    #              timeout = 10000)
    # assert False

    explorationCompression(baseGrammar, train,
                           testingTasks = test,
                           outputPrefix = "experimentOutputs/text",
                           evaluationTimeout = 0.0005,
                           **commandlineArguments(
#                               frontierSize = 10**4,
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
