from ec import explorationCompression, commandlineArguments, RegressionTask
from grammar import Grammar
from utilities import eprint, testTrainSplit
from makeTextTasks import makeTasks, delimiters
from textPrimitives import primitives

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

if __name__ == "__main__":
    tasks = makeTasks()
    for t in tasks:
        t.features = problemFeatures(t.examples)
    eprint("Generated",len(tasks),"tasks")

    test, train = testTrainSplit(tasks, 0.5)
    
    statistics = RegressionTask.standardizeTasks(train)
    featureExtractor = makeFeatureExtractor(statistics, tasks)

    baseGrammar = Grammar.uniform(primitives)
    explorationCompression(baseGrammar, train,
                           outputPrefix = "experimentOutputs/text",
                           **commandlineArguments(
                               frontierSize = 10**4,
                               iterations = 10,
                               a = 2,
                               featureExtractor = featureExtractor,
                               pseudoCounts = 10.0))
