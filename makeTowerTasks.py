from towerPrimitives import ttower
from utilities import *
from task import *

import math



TOWERCACHING = None
def initializeTowerCaching():
    global TOWERCACHING
    if False:
        from multiprocessing import Manager
        m = Manager()
        TOWERCACHING = m.dict()
    else:
        TOWERCACHING = {}

def getTowerCash():
    global TOWERCACHING
    return TOWERCACHING

class TowerTask(Task):
    tasks = []
    STABILITYTHRESHOLD = 0.5
    
    def __init__(self, _ = None, perturbation = 0,
                 maximumStaircase = 100,
                 maximumMass = 100,
                 minimumLength = 0,
                 minimumArea = 0,
                 minimumHeight = None):
        name = "; ".join("%s: %s"%(k,v) for k,v in locals().iteritems()
                         if not k in {"_","self"} )
        features = [perturbation,
                    float(maximumMass),
                    float(minimumHeight),
                    float(minimumLength),
                    float(minimumArea),
                    float(maximumStaircase)]
        super(TowerTask, self).__init__(name, ttower, [],
                                        features = features)

        self.maximumStaircase = maximumStaircase
        self.perturbation = perturbation
        self.minimumLength = minimumLength
        self.maximumMass = maximumMass
        self.minimumHeight = minimumHeight
        self.minimumArea = minimumArea

        TowerTask.tasks.append(self)

    @staticmethod
    def evaluateTower(tower, perturbation):
        global TOWERCACHING
        from towers.tower_common import TowerWorld
        
        key = (tuple(tower), perturbation)
        if key in TOWERCACHING:
            result = TOWERCACHING[key]
        else:
            def powerOfTen(n):
                if n <= 0: return False
                while True:
                    if n == 1: return True
                    if n % 10 != 0: return False
                    n = n/10
                
            if powerOfTen(len(TOWERCACHING)):
                eprint("Tower cache reached size",len(TOWERCACHING))
            w = TowerWorld()
            try: result = w.sampleStability(tower, perturbation, N = 15)
            except: result = None
            
            # except Exception as exception:
            #     eprint("exception",exception)
            #     eprint(perturbation, tower)
            #     raise exception                
            
            TOWERCACHING[key] = result
        return Bunch(result) if result is not None else result

    def logLikelihood(self, e, timeout=None):
        try: tower = runWithTimeout(lambda: e.evaluate([]), timeout)
        except: return NEGATIVEINFINITY

        
        mass = sum(w*h for _,w,h in tower)
        if mass > self.maximumMass: return NEGATIVEINFINITY

        tower = centerTower(tower)

        result = TowerTask.evaluateTower(tower, self.perturbation)
        if result is None: return NEGATIVEINFINITY
        
        if result.height < self.minimumHeight:
            return NEGATIVEINFINITY
        if result.staircase > self.maximumStaircase:
            return NEGATIVEINFINITY
        if result.stability < TowerTask.STABILITYTHRESHOLD:
            #eprint("stability")
            return NEGATIVEINFINITY
        if result.length < self.minimumLength:
            #eprint("len()", result.length)
            return NEGATIVEINFINITY
        if result.area < self.minimumArea:
            #eprint("area")
            return NEGATIVEINFINITY
        return 50.0*math.log(result.stability)

    def animateSolution(self, e):
        import os

        if isinstance(e, Program):
            tower = e.evaluate([])
        else:
            assert isinstance(e, list)
            tower = e

        os.system("python towers/visualize.py '%s' %f"%(tower, self.perturbation))

    def drawSolution(self,tower):
        from towers.tower_common import TowerWorld
        return TowerWorld().draw(tower)

def centerTower(t):
    x1 = max(x for x,_,_ in t )
    x0 = min(x for x,_,_ in t )
    c = float(x1 + x0)/2.
    return [ (x - c, w, h) for x,w,h in t ]
        
def makeTasks():
    STRONGPERTURBATION = 12
    MILDPERTURBATION = 8
    MASSES = [20,30,40]
    HEIGHT = [3,5,7]
    STAIRCASE = [10.5, 2.5]
    return [ TowerTask(maximumMass = float(m),
                       maximumStaircase = float(s),
                       minimumArea = float(a),
                       perturbation = float(p),
                       minimumLength = float(l),
                       minimumHeight = float(h))
             for m in MASSES
             for a in [1, 2.9, 5.8]
             for s in STAIRCASE 
             for l in [2.5, 5, 7.5]
             for p in [MILDPERTURBATION]#, STRONGPERTURBATION]
             for h in HEIGHT
             if not ((p == STRONGPERTURBATION and m == min(MASSES)) or \
                     (p == STRONGPERTURBATION and h == max(HEIGHT)))
    ]


