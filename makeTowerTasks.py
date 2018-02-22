from task import *

import math


class TowerTask(Task):
    RESULTCASH = {}
    def __init__(self, _ = None, perturbation = 0,
                 maximumBlocks = 100,
                 minimumHeight = None, minimumLength = None, maximumLength = None):
        name = "P: %f; H: %f; B: %d"%(perturbation, minimumHeight, maximumBlocks)
        super(TowerTask, self).__init__(name, tlist(tpair(tint,tbool)), [])

        self.perturbation = perturbation
        self.maximumBlocks = maximumBlocks
        self.minimumHeight = minimumHeight
        self.minimumLength = minimumLength
        self.maximumLength = maximumLength

    def logLikelihood(self, e, timeout = None):
        from towers.tower_common import TowerWorld
        
        tower = e.evaluate([])
        if len(tower) > self.maximumBlocks: return NEGATIVEINFINITY

        key = (tuple(tower), self.perturbation)
        if key in TowerTask.RESULTCASH: height, stabilities = TowerTask.RESULTCASH[key]
        else:
            w = TowerWorld()
            height, stabilities = w.sampleStability(tower, self.perturbation, N = 30)
            TowerTask.RESULTCASH[key] = (height, stabilities)

        if height < self.minimumHeight: return NEGATIVEINFINITY
        successProbability = float(sum(stabilities))/len(stabilities)
        if successProbability < 0.3: return NEGATIVEINFINITY

        return 20.0*math.log(successProbability)

    def animateSolution(self, e):
        import os

        tower = e.evaluate([])

        os.system("python towers/visualize.py '%s' %f"%(tower, self.perturbation))

        
        
def makeTasks():
    return [ TowerTask(maximumBlocks = 9,
                       perturbation = p,
                       minimumHeight = h,
                       minimumLength = minimum,
                       maximumLength = maximum)
             for p in [3,4]
             for h in [4,6,8]
             for minimum in [None] #+ range(1,2)
             for maximum in [None] #+ range(3,4)
    ]


