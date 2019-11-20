from dreamcoder.domains.tower.towerPrimitives import ttower, executeTower, _empty_tower, TowerState
from dreamcoder.domains.tower.tower_common import renderPlan
from dreamcoder.task import *


class SupervisedTower(Task):
    def __init__(self, name, program, mustTrain=False):
        if isinstance(program,str):
            try:
                program = parseTower(program)
            except:
                eprint("Parse failure:")
                eprint(program)
                assert False
            self.original = program
            plan = executeTower(program)
        elif isinstance(program,Program):
            self.original = program
            plan = executeTower(program)
        else:
            plan = program
        self.original = program
        state, self.plan = program.evaluate([])(_empty_tower)(TowerState())
        self.hand = state.hand
        super(SupervisedTower, self).__init__(name, arrow(ttower,ttower), [],
                                              features=[])
        self.specialTask = ("supervisedTower",
                            {"plan": self.plan})
        self.image = None
        self.handImage = None
        self.mustTrain = mustTrain

    def getImage(self, drawHand=False, pretty=False):
        if not drawHand:
            if not pretty:
                if self.image is not None: return self.image
                self.image = renderPlan(self.plan, pretty=pretty)
                return self.image
            else:
                return renderPlan(self.plan, pretty=True)
        else:
            if self.handImage is not None: return self.handImage
            self.handImage = renderPlan(self.plan,
                                        drawHand=self.hand,
                                        pretty=pretty)
            return self.handImage
                

    
    # do not pickle the image
    def __getstate__(self):
        return self.specialTask, self.plan, self.request, self.cache, self.name, self.examples
    def __setstate__(self, state):
        self.specialTask, self.plan, self.request, self.cache, self.name, self.examples = state
        self.image = None


    def animate(self):
        from pylab import imshow,show
        a = renderPlan(self.plan)
        imshow(a)
        show()

    @staticmethod
    def showMany(ts):
        from pylab import imshow,show
        a = montage([renderPlan(t.plan, pretty=True, Lego=True, resolution=256,
                                drawHand=False)
                     for t in ts]) 
        imshow(a)
        show()

    @staticmethod
    def exportMany(f, ts, shuffle=True, columns=None):
        import numpy as np
        
        ts = list(ts)
        if shuffle:
            assert all( t is not None for t in ts  )
            random.shuffle(ts)
        a = montage([renderPlan(t.plan, pretty=True, Lego=True, resolution=256) if t is not None \
                     else np.zeros((256,256,3))
                     for t in ts],
                    columns=columns)        
        import scipy.misc
        scipy.misc.imsave(f, a)
        

    def exportImage(self, f, pretty=True, Lego=True, drawHand=False):
        a = renderPlan(self.plan,
                       pretty=pretty, Lego=Lego,
                       drawHand=t.hand if drawHand else None)
        import scipy.misc
        scipy.misc.imsave(f, a)

    def logLikelihood(self, e, timeout=None):
        from dreamcoder.domains.tower.tower_common import centerTower
        yh = executeTower(e, timeout)
        if yh is not None and centerTower(yh) == centerTower(self.plan): return 0.
        return NEGATIVEINFINITY

    

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
    arches = [SupervisedTower("arch leg %d"%n,
                              "((for i %d v) (r 4) (for i %d v) (l 2) h)"%(n,n))
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
    offsetArches = [SupervisedTower("bridge (%d) of arch, spaced %d"%(n,l),
                                    """
                                    (for j %d
                                    (embed v (r 4) v (l 2) h )
                                    (r %d))
                                    """%(n,l),
                                    mustTrain=n == 3)
                    for n,l in [(3,7),(4,8)]]
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
    simpleLoops = [SupervisedTower("%s row %d, spacing %d"%(o,n,s),
                                   """(for j %d %s (r %s))"""%(n,o,s),
                                   mustTrain=True)
                   for o,n,s in [('h',4,7), ('v',5,3)] ]

    pyramids = []
    pyramids += [SupervisedTower("arch pyramid %d"%n,
                                 """((for i %d (for j i (embed v (r 4) v (l 2) h)) (r 6))
                                 (for i %d (for j (- %d i) (embed v (r 4) v (l 2) h)) (r 6)))"""%(n,n,n))
                for n in range(2,6) ]
    pyramids += [SupervisedTower("H pyramid %d"%n,
                                 """((for i %d (for j i h) (r 6))
                                 (for i %d (for j (- %d i) h) (r 6)))"""%(n,n,n))
                for n in range(4,6) ]
#     pyramids += [SupervisedTower("V pyramid %d"%n,
# """
# ((for i %d (for j i v) (r 2))
#  (for i %d (for j (- %d i) v) (r 2)))
# """%(n,n,n))
#                 for n in range(4,8) ]
#     pyramids += [SupervisedTower("V3 pyramid %d"%n,
# """
# ((for i %d (for j i v) (r 6))
#  (for i %d (for j (- %d i) v) (r 6)))
# """%(n,n,n))
#                  for n in range(4,8) ]
    pyramids += [SupervisedTower("H 1/2 pyramid %d"%n,
                                 """
(for i %d
  (r 6)
  (embed
    (for j i h (l 3))))
                                 """%n)
                for n in range(4,8) ]
    pyramids += [SupervisedTower("arch 1/2 pyramid %d"%n,
"""
(for i %d
  (r 6)
  (embed
    (for j i (embed v (r 4) v (l 2) h) (l 3))))
"""%n)
                for n in range(2,8) ]
    if False:
        pyramids += [SupervisedTower("V 1/2 pyramid %d"%n,
                                     """
    (for i %d
      (r 2)
      (embed
                                     (for j i v (l 1))))"""%(n))
                    for n in range(4,8) ]
    bricks = [SupervisedTower("brickwall, %dx%d"%(w,h),
                              """(for j %d
                              (embed (for i %d h (r 6)))
                              (embed (r 3) (for i %d h (r 6))))"""%(h,w,w))
              for w in range(3,7)
              for h in range(1,6) ]
    aqueducts = [SupervisedTower("aqueduct: %dx%d"%(w,h),
                                 """(for j %d
                                 %s (r 4) %s (l 2) h (l 2) v (r 4) v (l 2) h (r 4))"""%
                                 (w, "v "*h, "v "*h))
                 for w in range(4,8)
                 for h in range(3,6)
                 ]

    compositions = [SupervisedTower("%dx%d-bridge on top of %dx%d bricks"%(b1,b2,w1,w2),
                                    """
                                    ((for j %d
                                    (embed (for i %d h (r 6)))
                                    (embed (r 3) (for i %d h (r 6))))
                                    (r 1)
                                    (for j %d
                                    (for i %d 
                                    v (r 4) v (l 4)) (r 2) h 
                                    (r 4)))
                                    """%(w1,w2,w2,b1,b2))
                    for b1,b2,w1,w2 in [(5,2,4,5)]
                    ] + [
                        SupervisedTower("%d pyramid on top of %dx%d bricks"%(p,w1,w2),
                                        """
                                        ((for j %d
                                        (embed (for i %d h (r 6)))
                                        (embed (r 3) (for i %d h (r 6))))
                                        (r 1)
                                        (for i %d (for j i (embed v (r 4) v (l 2) h)) (r 6))
                                        (for i %d (for j (- %d i) (embed v (r 4) v (l 2) h)) (r 6)))
                                        """%(w1,w2,w2,p,p,p))
                        for w1,w2,p in [(2,5,2)]
                        ] + \
                        [
                            SupervisedTower("%d tower on top of %dx%d bricks"%(t,w1,w2),
                                            """
                                            ((for j %d
                                            (embed (for i %d h (r 6)))
                                            (embed (r 3) (for i %d h (r 6))))
                                            (r 6)
                                            %s (r 4) %s (l 2) h)
                                            """%(w1,w2,w2,
                                                 "v "*t, "v "*t))
                            for t,w1,w2 in [(4,1,3)] ]
                            
    
                     
    everything = arches + simpleLoops + Bridges + archesStacks + aqueducts + offsetArches + pyramids + bricks + staircase2 + staircase1 + compositions
    if False:
        for t in everything:
            delattr(t,'original')
    return everything

def makeOldSupervisedTasks():
    arches = [SupervisedTower("arch leg %d"%n,
                              "((for i %d v) (r 4) (for i %d v) (l 2) h)"%(n,n))
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
    offsetArches = [SupervisedTower("bridge (%d) of arch, spaced %d"%(n,l),
                               """
                               (for j %d
                                 v (r 4) v (l 2) h 
                                (r %d))
                               """%(n,l))
                    for n,l in [(3,7),(4,6)]]
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
    simpleLoops = [SupervisedTower("horizontal row %d, spacing %d"%(n,s),
                                   """(for j %d h (r %s))"""%(n,s))
                   for n,s in [(4,6),(5,7)] ]+\
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
#     pyramids += [SupervisedTower("V pyramid %d"%n,
# """
# ((for i %d (for j i v) (r 2))
#  (for i %d (for j (- %d i) v) (r 2)))
# """%(n,n,n))
#                 for n in range(4,8) ]
#     pyramids += [SupervisedTower("V3 pyramid %d"%n,
# """
# ((for i %d (for j i v) (r 6))
#  (for i %d (for j (- %d i) v) (r 6)))
# """%(n,n,n))
#                  for n in range(4,8) ]
    pyramids += [SupervisedTower("H 1/2 pyramid %d"%n,
                                 """
(for i %d
  (r 6)
  (embed
    (for j i h (l 3))))
                                 """%n)
                for n in range(4,8) ]
    pyramids += [SupervisedTower("arch 1/2 pyramid %d"%n,
"""
(for i %d
  (r 6)
  (embed
    (for j i (embed v (r 4) v (l 2) h) (l 3))))
"""%n)
                for n in range(2,8) ]
    if False:
        pyramids += [SupervisedTower("V 1/2 pyramid %d"%n,
                                     """
    (for i %d
      (r 2)
      (embed
                                     (for j i v (l 1))))"""%(n))
                    for n in range(4,8) ]
    bricks = [SupervisedTower("brickwall, %dx%d"%(w,h),
                              """(for j %d
                              (embed (for i %d h (r 6)))
                              (embed (r 3) (for i %d h (r 6))))"""%(h,w,w))
              for w in range(3,7)
              for h in range(1,6) ]
    aqueducts = [SupervisedTower("aqueduct: %dx%d"%(w,h),
                                 """(for j %d
                                 %s (r 4) %s (l 2) h (l 2) v (r 4) v (l 2) h (r 4))"""%
                                 (w, "v "*h, "v "*h))
                 for w in range(4,8)
                 for h in range(3,6)
                 ]

    compositions = [SupervisedTower("%dx%d-bridge on top of %dx%d bricks"%(b1,b2,w1,w2),
                                    """
                                    ((for j %d
                                    (embed (for i %d h (r 6)))
                                    (embed (r 3) (for i %d h (r 6))))
                                    (r 1)
                                    (for j %d
                                    (for i %d 
                                    v (r 4) v (l 4)) (r 2) h 
                                    (r 4)))
                                    """%(w1,w2,w2,b1,b2))
                    for b1,b2,w1,w2 in [(5,2,4,5)]
                    ] + [
                        SupervisedTower("%d pyramid on top of %dx%d bricks"%(p,w1,w2),
                                        """
                                        ((for j %d
                                        (embed (for i %d h (r 6)))
                                        (embed (r 3) (for i %d h (r 6))))
                                        (r 1)
                                        (for i %d (for j i (embed v (r 4) v (l 2) h)) (r 6))
                                        (for i %d (for j (- %d i) (embed v (r 4) v (l 2) h)) (r 6)))
                                        """%(w1,w2,w2,p,p,p))
                        for w1,w2,p in [(2,5,2)]
                        ] + \
                        [
                            SupervisedTower("%d tower on top of %dx%d bricks"%(t,w1,w2),
                                            """
                                            ((for j %d
                                            (embed (for i %d h (r 6)))
                                            (embed (r 3) (for i %d h (r 6))))
                                            (r 6)
                                            %s (r 4) %s (l 2) h)
                                            """%(w1,w2,w2,
                                                 "v "*t, "v "*t))
                            for t,w1,w2 in [(4,1,3)] ]
                            
    
                     
    everything = arches + simpleLoops + Bridges + archesStacks + aqueducts + offsetArches + pyramids + bricks + staircase2 + staircase1 + compositions
    if False:
        for t in everything:
            delattr(t,'original')
    return everything

def dSLDemo():
    DSL = {}
    bricks = Program.parse("(lambda (lambda (tower_loopM $0 (lambda (lambda (moveHand 3 (reverseHand (tower_loopM $3 (lambda (lambda (moveHand 6 (3x1 $0)))) $0))))))))")
    DSL["bricks"] = [ [bricks.runWithArguments([x,y + 4,_empty_tower,TowerState()])[1]
                       for y in range(6, 6 + 3*4, 3) ]
                      for x in [3,8] ]
    dimensionality = {}
    dimensionality["bricks"] = 2

    bridge = Program.parse("(lambda (lambda (tower_loopM $0 (lambda (lambda (#(lambda (#(lambda (lambda (lambda (tower_loopM $0 (lambda (lambda (1x3 (moveHand 4 ($3 $0))))) (moveHand 2 (3x1 $2)))))) $0 (lambda (reverseHand $0)))) (moveHand 4 $0) $3))))))")
    DSL["bridge"] = [ [bridge.runWithArguments([x,y,_empty_tower,TowerState()])[1]
                       for x in range(4,4 + 2*4,2) ]
                       for y in [4,9] ]
    dimensionality["bridge"] = 2

    staircase = Program.parse("(lambda (tower_loopM $0 (lambda (lambda (#(lambda (lambda (tower_loopM $1 (lambda (lambda (tower_embed (lambda (#(lambda (1x3 (moveHand 4 (1x3 (reverseHand (moveHand 2 (3x1 $0))))))) $0)) $0))) $0))) $1 (moveHand 6 $0))))))")
    DSL["staircase"] = [ staircase.runWithArguments([n,_empty_tower,TowerState()])[1]
                         for n in range(4,5 + 3) ]

    pyramid = Program.parse("(lambda (tower_loopM $0 (lambda (lambda (moveHand 6 (tower_embed (lambda (reverseHand ((lambda (lambda (tower_loopM $1 (lambda (lambda (moveHand $2 (1x3 (moveHand 2 (tower_embed (lambda (moveHand 2 (1x3 $0))) (3x1 $0)))))))))) $2 1 $0))) $0))))))")
    DSL["pyramid"] = [ pyramid.runWithArguments([n,_empty_tower,TowerState()])[1]
                       for n in range(4,5 + 3) ]

    towerArch = Program.parse("(lambda (lambda ((lambda ((lambda (lambda (lambda (tower_loopM $0 (lambda (lambda (1x3 (moveHand 4 ($3 $0))))) (moveHand 2 (3x1 $2)))))) $0 (lambda (reverseHand (1x3 $0))))) $0 $1)))")
    DSL["towerArch"] = [ towerArch.runWithArguments([n,_empty_tower,TowerState()])[1]
                       for n in range(4,5 + 3) ]

    images = {}
    for k,v in DSL.items():
        d = dimensionality.get(k,1)
        if d == 1:
            i = montageMatrix([[renderPlan(p, pretty=True, Lego=True) for p in v]])
        elif d == 2:
            i = montageMatrix([[renderPlan(p, pretty=True, Lego=True) for p in ps] for ps in v] )
        else: assert False

        images[k] = i

    return images
            
if __name__ == "__main__":
    from pylab import imshow,show
    from dreamcoder.domains.tower.tower_common import *
    
    ts = makeSupervisedTasks()
    print(len(ts),"total tasks")
    print("maximum plan length",max(len(f.plan) for f in ts ))
    print("maximum tower length",max(towerLength(f.plan) for f in ts ))
    print("maximum tower height",max(towerHeight(simulateWithoutPhysics(f.plan)) for f in ts ))
    SupervisedTower.exportMany("/tmp/every_tower.png",ts,shuffle=False)
    
    for j,t in enumerate(ts):
        t.exportImage("/tmp/tower_%d.png"%j,
                      drawHand=False)

    for k,v in dSLDemo().items():
        import scipy.misc
        scipy.misc.imsave(f"/tmp/tower_dsl_{k}.png", v)

    exampleTowers = [103,104,105,93,73,
                     50,67,35,43,106]
    SupervisedTower.exportMany("/tmp/tower_montage.png",
                               [ts[n] for n in exampleTowers ],
                               columns=5,
                               shuffle=False)
    assert False


    keywords = ["pyramid",
                "on top of",
                "arch 1/2 pyramid",
                "brickwall",
                "staircase",
                "bridge",
                "aqueduct",
                "spaced",
                "spaced",
                "arch stack"]
    for n in range(100):
        examples = []
        for kw in keywords:
            if kw == "on top of":
                examples = examples + list(filter(lambda t: kw in str(t), ts))
            else:
                examples.append(random.choice(list(filter(lambda t: kw in str(t), ts))))

        random.shuffle(examples)
        SupervisedTower.exportMany("/tmp/tower10_%d.png"%n,examples,
                                   columns=int(len(examples)/2))
        
        
        
