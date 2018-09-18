from towerPrimitives import ttower, executeTower
from utilities import *
from task import *

import math


class SupervisedTower(Task):
    def __init__(self, name, program):
        if isinstance(program,str):
            try:
                program = parseTower(program)
            except:
                eprint("Parse failure:")
                eprint(program)
                assert False
            self.original = program
            plan = program.evaluate([])(lambda s: (s,[]))(0)[1]
        else:
            plan = program
        super(SupervisedTower, self).__init__(name, arrow(ttower,ttower), [],
                                              features=[])
        self.specialTask = ("supervisedTower",
                            {"plan": plan})
        self.plan = plan
        self.image = None

    def getImage(self):
        from tower_common import fastRendererPlan
        if self.image is not None: return self.image

        self.image = fastRendererPlan(centerTower(self.plan))

        return self.image

    
    # do not pickle the image
    def __getstate__(self):
        return self.specialTask, self.plan, self.request, self.cache, self.name, self.examples
    def __setstate__(self, state):
        self.specialTask, self.plan, self.request, self.cache, self.name, self.examples = state
        self.image = None


    def animate(self):
        from tower_common import fastRendererPlan
        from pylab import imshow,show
        a = fastRendererPlan(centerTower(self.plan))
        imshow(a)
        show()

    @staticmethod
    def showMany(ts):
        from tower_common import fastRendererPlan
        from pylab import imshow,show
        a = montage([fastRendererPlan(centerTower(t.plan),pretty=True)
                         for t in ts]) 
        imshow(a)
        show()
        

    

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
        from tower_common import TowerWorld

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
    c = (x1 - x0) / 2 + x0
    return [(x - c, w, h) for x, w, h in t]

def towerLength(t):
    x1 = max(x for x, _, _ in t)
    x0 = min(x for x, _, _ in t)
    return x1 - x0


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

def parseTower(s):
    _13 = Program.parse("1x3")
    _31 = Program.parse("3x1")
    _r = Program.parse("right")
    _l = Program.parse("left")
    _addition = Program.parse("+")
    _subtraction = Program.parse("-")
    _lp = Program.parse("tower_loopM")
    _e = Program.parse("tower_embed")

    from sexpdata import loads, Symbol
    s = loads(s)
    def command(k, environment, continuation):
        if k == Symbol("1x3") or k == Symbol("v"): return Application(_13, continuation)
        if k == Symbol("3x1") or k == Symbol("h"): return Application(_31, continuation)
        assert isinstance(k,list)
        if k[0] == Symbol("r"): return Application(Application(_r, expression(k[1],environment)),continuation)
        if k[0] == Symbol("l"): return Application(Application(_l, expression(k[1],environment)),continuation)
        if k[0] == Symbol("for"):
            v = k[1]
            b = expression(k[2], environment)
            newEnvironment = [None, v] + environment
            body = block(k[3:], newEnvironment, Index(0))
            return Application(Application(Application(_lp,b),
                                           Abstraction(Abstraction(body))),
                               continuation)
        if k[0] == Symbol("embed"):
            body = block(k[1:], [None] + environment, Index(0))
            return Application(Application(_e,Abstraction(body)),continuation)
            
        assert False
    def expression(e, environment):
        for n, v in enumerate(environment):
            if e == v: return Index(n)

        if isinstance(e,int): return Program.parse(str(e))

        assert isinstance(e,list)
        if e[0] == Symbol('+'): return Application(Application(_addition, expression(e[1], environment)),
                                                   expression(e[2], environment))
        if e[0] == Symbol('-'): return Application(Application(_subtraction, expression(e[1], environment)),
                                                   expression(e[2], environment))
        assert False
        
    def block(b, environment, continuation):
        if len(b) == 0: return continuation
        return command(b[0], environment, block(b[1:], environment, continuation))

    try: return Abstraction(command(s, [], Index(0)))
    except: return Abstraction(block(s, [], Index(0)))

    
def makeSupervisedTasks():
    from towerPrimitives import _left,_right,_loop,_embed

    arches = [SupervisedTower("arch leg %d"%n,
                              "(%s (r 4) %s (l 2) h)"%("v "*n, "v "*n))
              for n in range(1,9)
    ]
    archesStacks = [SupervisedTower("arch stack %d"%n,
                                    """
                                    (for i %d 
                                    v (r 4) v (l 2) h (l 2))
                                    """%n)
                    for n in range(3,7) ]
    Bridges = [SupervisedTower("bridge (%d) of arch %d"%(n,l),
                               """
                               (for j %d
                                (for i %d 
                                 v (r 4) v (l 4)) (r 2) h 
                                (r 4))
                               """%(n,l))
               for n in range(2,8)
               for l in range(1,6)]
    Josh = [SupervisedTower("Josh (%d)"%n,
                            """(for i %d
                            h (l 2) v (r 2) v (r 2) v (l 2) h (r 6))"""%n)
            for n in range(1,7) ]
    
    staircase1 = [SupervisedTower("R staircase %d"%n,
"""
(for i %d (for j i
(embed v (r 4) v (l 2) h)) (r 6))
"""%(n))
                 for n in range(3,8) ]
    staircase2 = [SupervisedTower("L staircase %d"%n,
"""
(for i %d (for j i
(embed v (r 4) v (l 2) h)) (l 6))
"""%(n))
                 for n in range(3,8) ]
    simpleLoops = [SupervisedTower("horizontal row %d"%n,
                                   """(for j %d h (r 6))"""%n)
                   for n in [4,7] ]+\
                [SupervisedTower("vertical row %d"%n,
                                   """(for j %d v (r 2))"""%n)
                   for n in [3,6] ]+\
                [SupervisedTower("horizontal stack %d"%n,
                                   """(for j %d h)"""%n)
                   for n in range(5,8) ]+\
                [SupervisedTower("vertical stack %d"%n,
                                   """(for j %d v)"""%n)
                   for n in [5,7] ]
    pyramids = []
    pyramids += [SupervisedTower("arch pyramid %d"%n,
                                 """((for i %d (for j i (embed v (r 4) v (l 2) h)) (r 6))
                                 (for i %d (for j (- %d i) (embed v (r 4) v (l 2) h)) (r 6)))"""%(n,n,n))
                for n in range(2,6) ]
    pyramids += [SupervisedTower("H pyramid %d"%n,
                                 """((for i %d (for j i h) (r 6))
                                 (for i %d (for j (- %d i) h) (r 6)))"""%(n,n,n))
                for n in range(4,6) ]
    pyramids += [SupervisedTower("V pyramid %d"%n,
"""
((for i %d (for j i v) (r 2))
 (for i %d (for j (- %d i) v) (r 2)))
"""%(n,n,n))
                for n in range(4,8) ]
    pyramids += [SupervisedTower("V3 pyramid %d"%n,
"""
((for i %d (for j i v) (r 6))
 (for i %d (for j (- %d i) v) (r 6)))
"""%(n,n,n))
                 for n in range(4,8) ]
    pyramids += [SupervisedTower("H 1/2 pyramid %d"%n,
                                 """
(for i %d
  (r 6)
  (embed
    (for j i h (l 3))))
                                 """%n)
                for n in range(3,7) ]
    pyramids += [SupervisedTower("arch 1/2 pyramid %d"%n,
"""
(for i %d
  (r 6)
  (embed
    (for j i (embed v (r 4) v (l 2) h) (l 3))))
"""%n)
                for n in range(2,8) ]
    pyramids += [SupervisedTower("V 1/2 pyramid %d"%n,
                                 """
(for i %d
  (r 2)
  (embed
                                 (for j i v (l 1))))"""%(n))
                for n in range(3,8) ]
    bricks = [SupervisedTower("brickwall, %dx%d"%(w,h),
                              """(for j %d
                              (embed (for i %d h (r 6)))
                              (embed (r 3) (for i %d h (r 6))))"""%(h,w,w))
              for w in range(2,6)
              for h in range(1,6) ]
    aqueducts = [SupervisedTower("aqueduct: %dx%d"%(w,h),
                                 """(for j %d
                                 %s (r 4) %s (l 2) h (l 2) v (r 4) v (l 2) h (r 4))"""%
                                 (w, "v "*h, "v "*h))
                 for w in range(4,8)
                 for h in range(3,6)
                 ]
    everything = simpleLoops + arches + Bridges + archesStacks + aqueducts + Josh + pyramids + bricks + staircase2 + staircase1
    for t in everything:
        delattr(t,'original')
    return everything
if __name__ == "__main__":
    ts = makeSupervisedTasks()
    print(len(ts),"total tasks")
    print("maximum plan length",max(len(f.plan) for f in ts ))
    print("maximum tower length",max(towerLength(f.plan) for f in ts ))
    SupervisedTower.showMany(ts)
