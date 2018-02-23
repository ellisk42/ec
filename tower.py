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

        # Find the largest perturbation that this power can withstand
        for perturbation in sorted(set(TowerTask.POSSIBLEPERTURBATIONS), reverse = True):
            height, successProbability = TowerTask.evaluateTower(p, perturbation)
            if successProbability > TowerTask.STABILITYTHRESHOLD:         
                return [perturbation, maximumBlocks, height]

        return None


if __name__ == "__main__":
    g0 = Grammar.uniform(primitives)
    tasks = makeTasks()

    result = explorationCompression(g0, tasks,
                                    outputPrefix = "experimentOutputs/tower",
                                    solver = "python",
                                    **commandlineArguments(
                                        featureExtractor = TowerFeatureExtractor,
                                        CPUs = numberOfCPUs(),
                                        helmholtzRatio = 0.5,
                                        iterations = 5,
                                        a = 3,
                                        structurePenalty = 1,
                                        pseudoCounts = 10,
                                        topK = 10,
                                        maximumFrontier = 10**4))

    for t,frontier in result.taskSolutions.iteritems():
        if not frontier.empty:
            t.animateSolution(frontier.bestPosterior.program)
