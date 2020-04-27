
from dreamcoder.valueHead import *
from dreamcoder.program import *
from dreamcoder.type import tint
from dreamcoder.domains.tower.towerPrimitives import *
from dreamcoder.domains.tower.tower_common import simulateWithoutPhysics, centerTower


#TODO make abstract Primitives

"""
TODO:
- [ ] def of intTopState
- [ ] def of intTopPrimitive
- [X] seperate out intTop in primitives from intTop in hand state
- [ ] def of initial history state initialHist
- [X] write embed def
- [ ] write loop def, maybe with intTopPrimitive???
- [ ] deal with centerTower
- [ ] write taskViolatesAbstractState

"""
resolution = 256
initialHist = [(-resolution,0)]
_intTopState = (-resolution, resolution) #todo
_intTopPrimitive = (1, 8) #TODO


class ConvertSketchToAbstract:
    def __init__(self):
        pass
    def invented(self, e):
        return Invented(e.body.visit(self))
    def primitive(self, e ):
        name = e.name+"_abstract"
        return Primitive.GLOBALS[name]
    def index(self, e):
        return e
    def application(self, e):
        f = e.f.visit(self)
        x = e.x.visit(self)
        return Application(f, x)
    def abstraction(self, e):
        return Abstraction(e.body.visit(self))
    def hole(self, e):
        if e.tp == tint:
            return Primitive.GLOBALS["intTop"]
        elif e.tp == ttower:
            return Primitive.GLOBALS["towerTop"]
        else: assert False
    def execute(self, e):
        return e.visit(self)

def executeAbstractSketch(absSketch):
    #print("abstract Sketch:", absSketch)
    hand = absSketch.evaluate([])(lambda x: x)(AbstractTowerState(history=initialHist)) #TODO init history
    #print("hand.history", hand.history )
    return hand.history

def taskViolatesAbstractState(task, absState):
    
    def maxHeight(x, block):
        (x_,y_,w_,h_) = block
        x1_ = x_ - w_/2
        x2_ = x_ + w_/2
        if x1_ >= x or x >= x2_: return 0
        return y_ + h_//2

    #world is a (sorted) list of (x,y,w,h)
    world = simulateWithoutPhysics(centerTower(task.plan))
    print("world:", world)
    print("abstract state", absState)

    def checkViolationForParticularOffset(world, abstractState, offset):
        #abstractState = [(x + offset, y) for x, y in abstractState]
        world = [(x + offset, y, w, h) for x, y, w, h in world]
        for x in range(-resolution, resolution): #TODO
            #print(abstractState, x)
            if max([maxHeight(x,b) for b in world]) < heightAfter(abstractState, x):
                return True
        return False 

    lst = [checkViolationForParticularOffset(world, absState, offset) for offset in range(-resolution//2, resolution//2)] #TODO
    if all(lst):
        return True
    return False

class SymbolicAbstractTowers(BaseValueHead):
    def __init__(self):
        super(BaseValueHead, self).__init__()
        self.use_cuda = torch.cuda.is_available()
    def computeValue(self, sketch, task):

        absSketch = ConvertSketchToAbstract().execute(sketch)

        absState = executeAbstractSketch(absSketch)  #TODO

        if taskViolatesAbstractState(task, absState):
            return float('inf')

        return 0.

    def valueLossFromFrontier(self, frontier, g):
        if self.use_cuda:
            return torch.tensor([0.]).cuda()
        else: 
            return torch.tensor([0.])

class AbstractTowerState:
    def __init__(self, hand=(0,0), orientation=1, history=[]):
        self.history = history
        self.hand = hand
        self.orientation=orientation
    def __str__(self): return f"S(h={self.hand},o={self.orientation})"
    def __repr__(self): return str(self)

    def reverse(self):
        return AbstractTowerState(hand=self.hand, orientation=-1*self.orientation,
                    history=self.history )

    def move(self, n):
        if type(n) == int and self.orientation is not 0:
            newHand = (self.hand[0] + n*self.orientation, self.hand[1] + n*self.orientation)
        elif type(n) == int and self.orientation is 0:
            newHand = (self.hand[0] - n, self.hand[1] + n)

        elif n == _intTopPrimitive and self.orientation == 1:
            newHand = (self.hand[0] + 1, self.hand[1] + 9)
        elif n == _intTopPrimitive and self.orientation == -1:
            newHand = (self.hand[0] - 9, self.hand[1] - 1)
        elif n == _intTopPrimitive and self.orientation == 0:
            newHand = (self.hand[0] - 9, self.hand[1] + 9)
        else: assert False, "not allowed"

        return AbstractTowerState(hand=newHand , orientation=self.orientation,
                                         history=self.history )
    def recordBlock(self, b):
        #b is a block shape
        h, w = b

        rl, rh = self.hand
        newStuff = getNewBlock(rl, rh, w, h)
        if not newStuff:
            newHist = self.history
        else: 
            xl, xh, dh = newStuff
            newHist = updateMinHeight(self.history, xl, xh, dh)

        return AbstractTowerState(hand=self.hand, orientation=self.orientation,
                history=newHist)

    def topify(self):
        newHand = _intTopState #TODO find rang
        newOrientation = 0
        return AbstractTowerState(hand=newHand, orientation=newOrientation,
                history=self.history)



def _simpleLoop(n):
    if isinstance(n, int):
        def f(start, body, k):
            if start >= n: return k
            return body(start)(f(start + 1, body, k))
        return lambda b: lambda k: f(0,b,k)
    elif n == _intTopPrimitive:
        return lambda b: lambda k: lambda s: k(s.topify())
    else: assert 0


def _embed(body):
    def f(k):
        def g(hand):
            bodyHand = body(lambda x: x )(hand)
            # Record history if we are doing that
            if hand.history is not None:
                hand = AbstractTowerState(hand=hand.hand,
                                  orientation=hand.orientation,
                                  history=bodyHand.history)

            hand = k(hand)
            return hand
        return g
    return f


def _moveHand(n): return lambda k: lambda s: k(s.move(n))
def _reverseHand(k): return lambda s: k(s.reverse())


class AbsTowerContinuation(object):
    def __init__(self, x, w, h, name):
        self.x = x
        self.w = w*2
        self.h = h*2
        self.name = name

    def __call__(self, k):
        def f(hand):

            hand = hand.recordBlock( (self.w, self.h ) )
            hand = k(hand)
            return hand
        return f

# name, dimensions
blocks = {
    # "1x1": (1.,1.),
    # "2x1": (2.,1.),
    # "1x2": (1.,2.),
    "3x1": (3, 1),
    "1x3": (1, 3),
    #          "4x1": (4.,1.),
    #          "1x4": (1.,4.)
}


#TODO
abstractPrimitives = [
    Primitive('tower_loopM_abstract', arrow(tint, arrow(tint, ttower, ttower), ttower, ttower), _simpleLoop),
    Primitive("tower_embed_abstract", arrow(arrow(ttower,ttower), ttower, ttower), _embed), 
    ] + [Primitive(name+"_abstract", arrow(ttower,ttower), AbsTowerContinuation(0, w, h, name)) #TODO
     for name, (w, h) in blocks.items()] + \
    [Primitive(str(j)+'_abstract', tint, j) for j in range(1,9) ] + [
        Primitive("moveHand_abstract", arrow(tint, ttower, ttower), _moveHand),
        Primitive("reverseHand_abstract", arrow(ttower, ttower), _reverseHand),
        Primitive("towerTop", ttower, lambda x: x.topify()),

        Primitive("intTop", ttower, _intTopPrimitive) #TODO
    ]

def heightAfter(lst, p):
    #gets height at x, assuming things that end at x dont count, but thngs that start do
    preLst = [(x, y) for x,y in lst if x <= p ]
    return preLst[-1][1] 

def updateMinHeight(lst, xl, xh, dh):

    oldH = max ([heightAfter(lst, xl)] + [heightAfter(lst, p) for p, _ in lst if xl <= p < xh ] )
    
    postH = heightAfter(lst, xh)
    preLst = [(x, y) for x, y in lst if x < xl]
    postLst = [(x, y) for x, y in lst if x > xh]
    newLst = preLst + [(xl, oldH + dh), (xh, postH)] + postLst
    return newLst

def getNewBlock(rl, rh, w, h):
    if rh - rl > w: return None #todo
    xl = rh
    xh = rl + w
    return xl, xh, h

