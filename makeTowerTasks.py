from towerPrimitives import ttower, executeTower
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

class SupervisedTower(Task):
    def __init__(self, name, plan):
        self.original = plan
        plan = plan(lambda s: (s,[]))(0)[1]
        super(SupervisedTower, self).__init__(name, arrow(ttower,ttower), [],
                                              features=[])
        self.specialTask = ("supervisedTower",
                            {"plan": plan})
        self.plan = plan

    def animate(self):
        os.system("python towers/visualize.py '%s' 3"%(self.plan))

    

class TowerTask(Task):
    tasks = []
    STABILITYTHRESHOLD = 0.5

    def __init__(self, _=None, perturbation=0,
                 maximumStaircase=100,
                 maximumMass=100,
                 minimumLength=0,
                 minimumArea=0,
                 minimumOverpass=0,
                 minimumHeight=None):
        name = "; ".join("%s: %s" % (k, v) for k, v in locals().items()
                         if k not in {"_", "self", "__class__"})
        features = [perturbation,
                    float(maximumMass),
                    float(minimumHeight),
                    float(minimumLength),
                    float(minimumArea),
                    float(maximumStaircase),
                    float(minimumOverpass)]
        super(TowerTask, self).__init__(name, arrow(ttower,ttower), [],
                                        features=features)

        self.minimumOverpass = minimumOverpass
        self.maximumStaircase = maximumStaircase
        self.perturbation = perturbation
        self.minimumLength = minimumLength
        self.maximumMass = maximumMass
        self.minimumHeight = minimumHeight
        self.minimumArea = minimumArea

        self.specialTask = ("tower",
                            {"maximumStaircase": maximumStaircase,
                             "perturbation": perturbation,
                             "minimumLength": minimumLength,
                             "maximumMass": maximumMass,
                             "minimumHeight": minimumHeight,
                             "minimumArea": minimumArea,
                             "minimumOverpass": minimumOverpass})

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
                if n <= 0:
                    return False
                while True:
                    if n == 1:
                        return True
                    if n % 10 != 0:
                        return False
                    n = n / 10

            if powerOfTen(len(TOWERCACHING)):
                eprint("Tower cache reached size", len(TOWERCACHING))
                name = "experimentOutputs/towers%d.png" % len(TOWERCACHING)
                #exportTowers(list(set([ _t for _t,_ in TOWERCACHING.keys()])), name)
                eprint("Exported towers to image", name)
            w = TowerWorld()
            try:
                result = w.sampleStability(tower, perturbation, N=15)
            except BaseException:
                result = None

            # except Exception as exception:
            #     eprint("exception",exception)
            #     eprint(perturbation, tower)
            #     raise exception

            TOWERCACHING[key] = result
        return Bunch(result) if result is not None else result

    def logLikelihood(self, e, timeout=None):
        if timeout is not None:
            def timeoutCallBack(_1, _2): raise EvaluationTimeout()
            signal.signal(signal.SIGVTALRM, timeoutCallBack)
            signal.setitimer(signal.ITIMER_VIRTUAL, timeout)

        try:
            tower = executeTower(e)
            if timeout is not None:
                signal.signal(signal.SIGVTALRM, lambda *_: None)
                signal.setitimer(signal.ITIMER_VIRTUAL, 0)
        except EvaluationTimeout:
            return NEGATIVEINFINITY
        except BaseException:
            if timeout is not None:
                signal.signal(signal.SIGVTALRM, lambda *_: None)
                signal.setitimer(signal.ITIMER_VIRTUAL, 0)
            return NEGATIVEINFINITY

        mass = sum(w * h for _, w, h in tower)
        if mass > self.maximumMass:
            return NEGATIVEINFINITY

        tower = centerTower(tower)

        result = TowerTask.evaluateTower(tower, self.perturbation)
        if result is None:
            return NEGATIVEINFINITY

        if result.height < self.minimumHeight:
            return NEGATIVEINFINITY
        if result.staircase > self.maximumStaircase:
            return NEGATIVEINFINITY
        if result.stability < TowerTask.STABILITYTHRESHOLD:
            # eprint("stability")
            return NEGATIVEINFINITY
        if result.length < self.minimumLength:
            #eprint("len()", result.length)
            return NEGATIVEINFINITY
        if result.area < self.minimumArea:
            # eprint("area")
            return NEGATIVEINFINITY
        if result.overpass < self.minimumOverpass:
            return NEGATIVEINFINITY
        return 50.0 * math.log(result.stability)

    def animateSolution(self, e):
        import os

        if isinstance(e, Program):
            tower = executeTower(e)
        else:
            assert isinstance(e, list)
            tower = e

        os.system(
            "python towers/visualize.py '%s' %f" %
            (tower, self.perturbation))

    def drawSolution(self, tower):
        from towers.tower_common import TowerWorld
        return TowerWorld().draw(tower)


def centerTower(t):
    x1 = max(x for x, _, _ in t)
    x0 = min(x for x, _, _ in t)
    c = float(x1 - x0) / 2. + x0
    return [(x - c, w, h) for x, w, h in t]


def makeTasks():
    """ideas:
    House (enclose some minimum area)
    Bridge (be able to walk along it for some long distance, and also have a certain minimum height; enclosing elevated area at a certain height)
    Overhang (like a porch)
    Overpass (have a large hole)"""
    
    MILDPERTURBATION = 2
    MASSES = [500]
    HEIGHT = [1.9, 6, 10]
    STAIRCASE = [10.5, 2.5, 1.5]
    OVERPASS = [2.9,5.8]
    LENGTHS = [2, 5, 8]
    AREAS = [1, 2.9, 5.8, 11.6]
    return [TowerTask(maximumMass=float(m),
                      maximumStaircase=float(s),
                      minimumArea=float(a),
                      perturbation=float(p),
                      minimumLength=float(l),
                      minimumHeight=float(h),
                      minimumOverpass=float(o))
            for o in OVERPASS 
            for m in MASSES
            for a in AREAS
            for s in STAIRCASE
            for l in LENGTHS
            for p in [MILDPERTURBATION]
            for h in HEIGHT
            if o <= a
            ]
def makeSupervisedTasks():
    from towerPrimitives import epsilon,TowerContinuation,xOffset,_left,_right,_loop
    w,h = 2,1
    _21 = TowerContinuation(xOffset(w, h), w - 2*epsilon, h - epsilon)
    w,h = 1,2
    _12 = TowerContinuation(xOffset(w, h), w - 2*epsilon, h - epsilon)
    w,h = 1,3
    _13 = TowerContinuation(xOffset(w, h), w - 2*epsilon, h - epsilon)
    w,h = 3,1
    _31 = TowerContinuation(xOffset(w, h), w - 2*epsilon, h - epsilon)
    r = lambda n,k: _right(n)(k)
    l = lambda n,k: _left(n)(k)

    arches = [SupervisedTower("arch leg 1",lambda z: \
               _13(r(2,_13(l(1,_31(z)))))),
              SupervisedTower("arch leg 2",lambda z: \
               _13(_13(r(2,_13(_13(l(1,_31(z)))))))),
              SupervisedTower("arch leg 3",lambda z: \
               _13(_13(_13(r(2,_13(_13(_13(l(1,_31(z)))))))))),
              SupervisedTower("arch leg 4",lambda z: \
               _13(_13(_13(_13(r(2,_13(_13(_13(_13(l(1,_31(z))))))))))))
    ]
    Bridges = [SupervisedTower("bridge (%d) of %s"%(n,a.name),
                lambda z: _loop(n)(lambda i: a.original(r(2,z)))(z))
               for n in range(2,5)
               for a in arches]
    Josh = [SupervisedTower("Josh (%d)"%n,
                            lambda z: _loop(n)(lambda i: _31(_13(l(1,_13(r(2,_13(l(1,(_31(r(3,z)))))))))))(z))
            for n in range(1,5) ]
    staircase = [SupervisedTower("staircase %d"%n,
                                 lambda z: _loop(n)(lambda i: _loop(i)(lambda j: _13(r(2,_13(l(1,_31(l(1,z)))))))(r(3,z)))(z))
                 for n in range(2,5) ]
    simpleLoops = [SupervisedTower("horizontal row %d"%n,
                                   lambda z: _loop(n)(lambda i: _31(r(3,z)))(z))
                   for n in range(3,5) ]+\
                [SupervisedTower("vertical row %d"%n,
                                   lambda z: _loop(n)(lambda i: _13(r(1,z)))(z))
                   for n in range(3,5) ]+\
                [SupervisedTower("horizontal stack %d"%n,
                                   lambda z: _loop(n)(lambda i: _31(z))(z))
                   for n in range(3,5) ]+\
                [SupervisedTower("vertical stack %d"%n,
                                   lambda z: _loop(n)(lambda i: _13(z))(z))
                   for n in range(3,5) ]
    everything = simpleLoops + staircase + Josh + arches + Bridges
    for t in everything:
        delattr(t,'original')
    return everything
if __name__ == "__main__":
    for t in makeSupervisedTasks(): t.animate()
