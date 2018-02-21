from task import *

import math


class TowerTask(Task):
    def __init__(self, name, _ = None, perturbations = [],
                 minimumHeight = None, minimumLength = None, maximumLength = None):
        super(TowerTask, self).__init__(name, tlist(tpair(tint,tbool)), [])

        self.perturbations = perturbations
        self.minimumHeight = minimumHeight
        self.minimumLength = minimumLength
        self.maximumLength = maximumLength

    def logLikelihood(self, e, timeout = None):
        from towers.simulator import simulateTower, towerLength
        
        tower = e.evaluate([])

        l = towerLength(tower)
        if self.minimumLength is not None and l < self.minimumLength: return NEGATIVEINFINITY
        if self.maximumLength is not None and l > self.maximumLength: return NEGATIVEINFINITY

        result = simulateTower(tower, self.perturbations)
        if result is None: return NEGATIVEINFINITY

        if any( s <= 0.1 for s in result.stability ): return NEGATIVEINFINITY
        if result.height < self.minimumHeight: return NEGATIVEINFINITY

        return sum( math.log(s/100.) for s in result.stability )
        
def makeTasks():
    return [ TowerTask("P: %f; H: %f; max W: %s; min W: %s"%(p,h, minimum, maximum),
                       perturbations = [p],
                       minimumHeight = h,
                       minimumLength = minimum,
                       maximumLength = maximum)
             for p in [0.,1.,2.]
             for h in range(0,9,3)
             for minimum in [None] + range(1,2)
             for maximum in [None] + range(3,4) ]


