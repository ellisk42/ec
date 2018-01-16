from ec import *
from makeListProblems import makeTasks
from listPrimitives import primitives


if __name__ == "__main__":
    tasks = makeTasks()
    print "Got",len(tasks),"list tasks"

    explorationCompression(primitives, tasks,
                           **commandlineArguments(frontierSize = 10**4,
                                                  iterations = 5,
                                                  pseudoCounts = 10.0))

