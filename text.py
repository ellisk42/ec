from ec import *
from utilities import eprint
from makeTextTasks import makeTasks, delimiters
from textPrimitives import primitives

def stringFeatures(s):
    return [len(s)] + [sum(x == d for x in s ) for d in delimiters ] + [sum(x.upper() == x for x in s )]
def problemFeatures(task):
    inputFeatures = []
    outputFeatures = []
    for x,y in task.examples:
        inputFeatures.append(stringFeatures(x))
        outputFeatures.append(stringFeatures(y))
    n = float(len(task.examples))
    inputFeatures = map(lambda *a: sum(a)/n, *inputFeatures)
    outputFeatures = map(lambda *a: sum(a)/n, *outputFeatures)
    return inputFeatures + outputFeatures


if __name__ == "__main__":
    tasks = makeTasks()
    for t in tasks:
        t.features = problemFeatures(t)
    eprint("Generated",len(tasks),"tasks")

    # e = Program.parse("(lambda (join '@' (map (lambda emptyString) (split ',' $0))))")
    # print e
    # print e.evaluate([])("test,this,here")
    # assert False

    explorationCompression(primitives, tasks,
                           outputPrefix = "experimentOutputs/text",
                           **commandlineArguments(
                               frontierSize = 10**4,
                               iterations = 10,
                               a = 2,
                               pseudoCounts = 10.0))
