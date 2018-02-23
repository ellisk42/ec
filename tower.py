from ec import *

from towerPrimitives import primitives
from makeTowerTasks import *

import os
import random

class TowerFeatureExtractor(HandCodedFeatureExtractor):
    def _featuresOfProgram(self, p, _):
        # [perturbation, blocks, height]
        p = p.evaluate([])
        maximumBlocks = len(p)

        perturbation = random.choice(TowerTask.POSSIBLEPERTURBATIONS)
        height, successProbability = TowerTask.evaluateTower(p, perturbation)

        if successProbability < TowerTask.STABILITYTHRESHOLD: return None
        
        return [perturbation, maximumBlocks, height]


if __name__ == "__main__":
    g0 = Grammar.uniform(primitives)
    tasks = makeTasks()

    result = explorationCompression(g0, tasks,
                                    outputPrefix = "experimentOutputs/tower",
                                    solver = "python",
                                    **commandlineArguments(
                                        featureExtractor = TowerFeatureExtractor,
                                        CPUs = numberOfCPUs(),
                                        iterations = 5,
                                        pseudoCounts = 20,
                                        topK = 10,
                                        maximumFrontier = 10**4))

    for t,frontier in result.taskSolutions.iteritems():
        if not frontier.empty:
            t.animateSolution(frontier.bestPosterior.program)
