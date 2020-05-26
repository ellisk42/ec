
from dreamcoder.valueHead import *
from dreamcoder.program import *
from dreamcoder.type import tint
from dreamcoder.domains.tower.towerPrimitives import *
from dreamcoder.domains.tower.tower_common import simulateWithoutPhysics, centerTower
import numpy as np

#TODO make abstract Primitives

"""
TODO:
- [ ] def of intTopState
- [ ] def of intTopPrimitive
- [X] seperate out intTop in primitives from intTop in hand state
- [ ] def of initial history state initialHist
- [X] write embed def
- [X] write loop def, maybe with intTopPrimitive???
- [ ] deal with centerTower
- [ ] write taskViolatesAbstractState
- [ ] check that things make sense rigourously, in terms of resolution, which offsets to check, etc

"""
resolution = 256
initialHist = [(-resolution-1,0)]
_intTopState = (-resolution, resolution) #todo
_intTopPrimitive = (1, 8) #TODO
_intMax = 8
_intMin = 1

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
    hand = absSketch.evaluate([])(lambda x: x)(AbstractTowerState(history=initialHist)) #TODO init history
    return hand

def taskViolatesAbstractState(task, absWorld):
    
    def maxHeightAfter(x, block):
        (x_,y_,w_,h_) = block
        x1_ = x_ - w_/2
        x2_ = x_ + w_/2
        if x1_ > x or x >= x2_: return 0
        return y_ + h_//2

    #world is a (sorted) list of (x,y,w,h)
    world = simulateWithoutPhysics(centerTower(task.plan))
    #print("world:", world)
    #print("abstract state", absWorld)

    def checkViolationForParticularOffset(world, abstractState, offset):
        #abstractState = [(x + offset, y) for x, y in abstractState]
        world = [(x + offset, y, w, h) for x, y, w, h in world]
        for x in range(-resolution, resolution): #TODO
            #print(abstractState, x)
            if max([maxHeightAfter(x,b) for b in world]) < heightAfter(abstractState, x):
                return True
        return False 

    lst = [checkViolationForParticularOffset(world, absWorld, offset) for offset in range(-resolution//2, resolution//2)] #TODO
    if all(lst):
        return True
    return False

class SymbolicAbstractTowers(BaseValueHead):
    def __init__(self):
        super(BaseValueHead, self).__init__()
        self.use_cuda = torch.cuda.is_available()
    def computeValue(self, sketch, task):

        absSketch = ConvertSketchToAbstract().execute(sketch)
        absState = executeAbstractSketch(absSketch) #gets he history

        if taskViolatesAbstractState(task, absState.history):
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

    def __eq__(self, other):
        return self.history == other.history and (self.hand == other.hand and self.orientation == other.orientation)

    def __hash__(self):
        return hash(self.hand) + hash(self.orientation) + hash(tuple(self.history))

    def reverse(self):
        return AbstractTowerState(hand=self.hand, orientation=-1*self.orientation,
                    history=self.history )

    def move(self, n):
        if type(n) == int and self.orientation is not 0:
            newHand = (self.hand[0] + n*self.orientation, self.hand[1] + n*self.orientation)
        elif type(n) == int and self.orientation is 0:
            newHand = (self.hand[0] - n, self.hand[1] + n)

        elif n == _intTopPrimitive and self.orientation == 1:
            newHand = (self.hand[0] + _intMin, self.hand[1] + _intMax)
        elif n == _intTopPrimitive and self.orientation == -1:
            newHand = (self.hand[0] - _intMax, self.hand[1] - _intMin)
        elif n == _intTopPrimitive and self.orientation == 0:
            newHand = (self.hand[0] - _intMax, self.hand[1] + _intMax)
        else: assert False, "not allowed"

        return AbstractTowerState(hand=newHand , orientation=self.orientation,
                                         history=self.history )
    def recordBlock(self, b):
        #b is a block shape
        w, h = b

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

    def asymmetricUnion(self, other): 
        #for loops
        #slight hack is that we take the history of self, because we know there is less in it
        if self.orientation == other.orientation:
            newOrientation = self.orientation
        else:
            newOrientation = 0
        newHand = (min(self.hand[0], other.hand[0]), max(self.hand[1], other.hand[1]))

        newHist = self.history
        return AbstractTowerState(hand=newHand, orientation=newOrientation,
                    history=newHist)



def _simpleLoop(n):
    if isinstance(n, int):
        def f(start, body, k):
            if start >= n: return k
            return body(start)(f(start + 1, body, k))
        return lambda b: lambda k: f(0,b,k)

    elif n == _intTopPrimitive:
        def f(start, body, k):
            if start >= _intTopPrimitive[1]: return k

            if start == 0: 
                return body(start)(f(start + 1, body, k))

            def newBody(startInt):
                def g(k):
                    def h(state):
                        return k( state.asymmetricUnion( body(startInt) (lambda x: x) (state) ))
                    return h
                return g

            return newBody(start)(f(start + 1, body, k))
            #return g( f(start+1, body, k))

        return lambda b: lambda k: f(0,b,k) 
        #return lambda b: lambda k: lambda s: k(s.topify())
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

        Primitive("intTop", tint, _intTopPrimitive) #TODO
    ]

def renderAbsTowerHist(state, renderHand=False):
    lst = state.history
    a = np.zeros((resolution, resolution, 3))
    
    green = [0, 0.5, 0]
    red = [0.5, 0, 0]
    yellow = [0.5, 0.5, 0]
    if state.orientation == 1:
        handColor = green
    elif state.orientation == -1:
        handColor = red
    else: handColor = yellow

    #print(state.hand, state.history)

    for x in range(-resolution+1, resolution-1, 2):
        if renderHand:

            if state.hand[0]-1 <= x and x <= state.hand[1]-1:
                for i in range(resolution):
                    a[i, resolution//2  + (x-1)//2, :] = handColor

        h = heightAfter(lst, x)
        a[resolution - h//2:, resolution//2  + (x-1)//2 , 1:] = 1.
    return a


def heightAfter(lst, p):
    #gets height at x, assuming things that end at x dont count, but thngs that start do
    preLst = [(x, y) for x,y in lst if x <= p ]
    if preLst:
        return preLst[-1][1] 
    else: return 0 #since I can't tell, assume 0?

def updateMinHeight(lst, xl, xh, dh):

    oldH = max ([heightAfter(lst, xl)] + [heightAfter(lst, p) for p, _ in lst if xl <= p < xh ] )
    
    postH = heightAfter(lst, xh)

    preLst = [(x, y) for x, y in lst if x < xl]
    postLst = [(x, y) for x, y in lst if x > xh]

    newLst = preLst + [(xl, oldH + dh), (xh, postH)] + postLst

    return newLst

def getNewBlock(rl, rh, w, h):
    if rh - rl > w: return None #todo
    xl = rh - w/2
    xh = rl + w/2
    return xl, xh, h

"""
         (lambda (#(lambda (lambda (#(lambda (lambda (lambda (tower_loopM $1 (lambda (lambda (#(lambda (lambda (lambda (tower_loopM $1 (lambda (lambda (1x3 (moveHand 4 ($2 $0))))) (moveHand 2 (3x1 $2)))))) (moveHand $2 $0) $4 (lambda (reverseHand $0))))))))) $0 $1 4))) 5 8 (moveHand 2 $0)))
"""

