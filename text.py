from ec import *
from makeStringTransformationProblems import makeTasks
from textPrimitives import primitives


if __name__ == "__main__":
    tasks = makeTasks()
    for t in tasks:
        t.features = [1]
        t.cache = False
    print "Generated",len(tasks),"tasks"

    explorationCompression(primitives, tasks,
                           **commandlineArguments(
                               frontierSize = 10**4,
                               iterations = 3,
                               pseudoCounts = 10.0))
