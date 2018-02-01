from ec import explorationCompression, commandlineArguments, RegressionTask
from grammar import Grammar
from utilities import eprint, testTrainSplit, numberOfCPUs
from makeTextTasks import makeTasks, delimiters
from textPrimitives import primitives
from program import *
from recognitionModel import *

import random

def stringFeatures(s):
    return [len(s)] + [sum(x == d for x in s ) for d in delimiters ] + [sum(x.upper() == x for x in s )]
def problemFeatures(examples):
    inputFeatures = []
    outputFeatures = []
    for (x,),y in examples:
        inputFeatures.append(stringFeatures(x))
        outputFeatures.append(stringFeatures(y))
    n = float(len(examples))
    inputFeatures = map(lambda *a: sum(a)/n, *inputFeatures)
    outputFeatures = map(lambda *a: sum(a)/n, *outputFeatures)
    return inputFeatures + outputFeatures

def makeFeatureExtractor((averages, deviations), tasks):
    # Try to make inputs that kind of look like the training inputs
    numberOfInputs = sum(len(t.examples) for t in tasks )/len(tasks)
    inputs = [ x for t in tasks for x,_ in t.examples ]
    def featureExtractor(program, tp):
        e = program.evaluate([])
        examples = []
        shuffledInputs = list(inputs)
        random.shuffle(shuffledInputs)
        for x in shuffledInputs:
            try:
                y = e(x[0])
                examples.append((x,y))
            except: continue
            if len(examples) >= numberOfInputs:
                # eprint("program",program)
                # for (x,),y in examples:
                #     eprint(x,"\t",y)
                # eprint()
                return RegressionTask.standardizeFeatures(averages, deviations, problemFeatures(examples))
        #eprint("Could only make %d examples for %s"%(len(examples), program))
        return None
    return featureExtractor

class FeatureExtractor(RecurrentFeatureExtractor):
    def __init__(self, inputs, numberOfExamples):
        self.numberOfExamples = numberOfExamples
        self.inputs = inputs
        lexicon = set([
            for t in tasks
            for (x,),y in t.examples
            for c in x + y ] + ["START","ENDING"])
        super(FeatureExtractor, self).__init__(lexicon, bidirectional = True)

    def taskFeatures(self, task):
        examples = [ ([x],y)
                     for (x,),y in task.examples ]
        return self.forward(examples)

    def programFeatures(self, program, t):
        assert t == arrow(tstr,tstr)
        random.shuffle(self.inputs)

        e = program.evaluate([])
        examples = []
        for x in self.inputs:
            try:
                y = e(x)
                examples.append(([x],y))
            except: continue
            if len(examples) >= self.numberOfExamples:
                return self(examples)
        
    

if __name__ == "__main__":
    tasks = makeTasks()
    for t in tasks:
        t.features = problemFeatures(t.examples)
    eprint("Generated",len(tasks),"tasks")

    test, train = testTrainSplit(tasks, 1.)

    statistics = RegressionTask.standardizeTasks(train)
    featureExtractor = makeFeatureExtractor(statistics, tasks)

    baseGrammar = Grammar.uniform(primitives)

    explorationCompression(baseGrammar, train,
                           outputPrefix = "experimentOutputs/text",
                           **commandlineArguments(
                               frontierSize = 10**4,
                               iterations = 10,
                               helmholtzRatio = 0.5,
                               topK = 2,
                               maximumFrontier = 1000,
                               structurePenalty = 5.,
                               a = 3,
                               activation = "relu",
                               CPUs = numberOfCPUs(),
                               featureExtractor = featureExtractor,
                               pseudoCounts = 10.0))
