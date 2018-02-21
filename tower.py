from ec import *

from towerPrimitives import primitives
from makeTowerTasks import *

import os




if __name__ == "__main__":
    g0 = Grammar.uniform(primitives)
    tasks = makeTasks()

    explorationCompression(g0, tasks,
                           outputPrefix = "experimentOutputs/tower",
                           solver = "python",
                           **commandlineArguments(
                               iterations = 5,
                               topK = 10,
                               maximumFrontier = 10**4))
