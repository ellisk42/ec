from towerPrimitives import ttower

from task import *

import math


class TowerTask(Task):
    RESULTCASH = {}
    tasks = []
    STABILITYTHRESHOLD = 0.5
    
    def __init__(self, _ = None, perturbation = 0,
                 maximumMass = 100,
                 minimumLength = 0,
                 minimumArea = 0,
                 minimumHeight = None):
        name = "; ".join("%s: %s"%(k,v) for k,v in locals() .iteritems()
                         if not k in {"_","self"} )
        features = [perturbation,
                    float(maximumMass),
                    float(minimumHeight),
                    float(minimumLength),
                    float(minimumArea)]
        super(TowerTask, self).__init__(name, ttower, [],
                                        features = features)

        self.perturbation = perturbation
        self.minimumLength = minimumLength
        self.maximumMass = maximumMass
        self.minimumHeight = minimumHeight
        self.minimumArea = minimumArea

        TowerTask.tasks.append(self)

    @staticmethod
    def evaluateTower(tower, perturbation):
        from towers.tower_common import TowerWorld
        
        key = (tuple(tower), perturbation)
        if key in TowerTask.RESULTCASH: result = TowerTask.RESULTCASH[key]
        else:
            w = TowerWorld()
            result = w.sampleStability(tower, perturbation, N = 30)
            TowerTask.RESULTCASH[key] = result
        return result

    def logLikelihood(self, e, timeout = None):
        tower = e.evaluate([])
        mass = sum(w*h for _,w,h in tower)
        if mass > self.maximumMass: return NEGATIVEINFINITY

        result = TowerTask.evaluateTower(tower, self.perturbation)
        
        if result.height < self.minimumHeight: return NEGATIVEINFINITY
        if result.stability < TowerTask.STABILITYTHRESHOLD: return NEGATIVEINFINITY
        if result.length < self.minimumLength: return NEGATIVEINFINITY
        if result.area < self.minimumArea: return NEGATIVEINFINITY

        return 50.0*math.log(result.stability)

    def animateSolution(self, e):
        import os

        tower = e.evaluate([])

        os.system("python towers/visualize.py '%s' %f"%(tower, self.perturbation))

        
        
def makeTasks():
    return [ TowerTask(maximumMass = float(m),
                       minimumArea = float(a),
                       perturbation = float(p),
                       minimumLength = float(l),
                       minimumHeight = float(h))
             for m in [10,20,30]
             for a in [1, 2.9, 5.8]
             for l in [0, 5]
             for p in [10, 15]
             for h in [4,5,6,7]
    ]


