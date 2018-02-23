from task import *

import math


class TowerTask(Task):
    RESULTCASH = {}
    POSSIBLEPERTURBATIONS = []
    STABILITYTHRESHOLD = 0.5
    
    def __init__(self, _ = None, perturbation = 0,
                 maximumBlocks = 100,
                 minimumHeight = None):
        name = "P: %f; H: %f; B: %d"%(perturbation, minimumHeight, maximumBlocks)
        features = [perturbation, float(maximumBlocks), float(minimumHeight)]
        super(TowerTask, self).__init__(name, tlist(tpair(tint,tbool)), [],
                                        features = features)

        self.perturbation = perturbation
        self.maximumBlocks = maximumBlocks
        self.minimumHeight = minimumHeight

        TowerTask.POSSIBLEPERTURBATIONS.append(perturbation)

    @staticmethod
    def evaluateTower(tower, perturbation):
        from towers.tower_common import TowerWorld
        
        key = (tuple(tower), perturbation)
        if key in TowerTask.RESULTCASH: height, stabilities = TowerTask.RESULTCASH[key]
        else:
            w = TowerWorld()
            height, stabilities = w.sampleStability(tower, perturbation, N = 30)
            TowerTask.RESULTCASH[key] = (height, stabilities)
        return height, stabilities

    def logLikelihood(self, e, timeout = None):
        tower = e.evaluate([])
        if len(tower) > self.maximumBlocks: return NEGATIVEINFINITY

        height, successProbability = TowerTask.evaluateTower(tower, self.perturbation)
        
        if height < self.minimumHeight: return NEGATIVEINFINITY
        if successProbability < TowerTask.STABILITYTHRESHOLD: return NEGATIVEINFINITY

        return 50.0*math.log(successProbability)

    def animateSolution(self, e):
        import os

        tower = e.evaluate([])

        os.system("python towers/visualize.py '%s' %f"%(tower, self.perturbation))

        
        
def makeTasks():
    return [ TowerTask(maximumBlocks = m,
                       perturbation = p,
                       minimumHeight = h)
             for m in [7,15] 
             for p in [2,3,4]
             for h in [4,6,8,10]
    ]


