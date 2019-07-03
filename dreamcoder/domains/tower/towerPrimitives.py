from dreamcoder.program import *


class TowerState:
    def __init__(self, hand=0, orientation=1, history=None):
        # List of (State|Block)
        self.history = history
        self.hand = hand
        self.orientation = orientation
    def __str__(self): return f"S(h={self.hand},o={self.orientation})"
    def __repr__(self): return str(self)
    def left(self, n):
        return TowerState(hand=self.hand - n, orientation=self.orientation,
                          history=self.history if self.history is None \
                          else self.history + [self])
    def right(self, n): return TowerState(hand=self.hand + n, orientation=self.orientation,
                                          history=self.history if self.history is None \
                                          else self.history + [self])
    def reverse(self): return TowerState(hand=self.hand, orientation=-1*self.orientation,
                                         history=self.history if self.history is None \
                                         else self.history + [self])
    def move(self, n): return TowerState(hand=self.hand + n*self.orientation, orientation=self.orientation,
                                         history=self.history if self.history is None \
                                         else self.history + [self])

    def recordBlock(self, b):
        if self.history is None: return self
        return TowerState(hand=self.hand,
                          orientation=self.orientation,
                          history=self.history + [b])
        

def _empty_tower(h): return (h,[])
def _left(d):
    return lambda k: lambda s: k(s.left(d))
def _right(d):
    return lambda k: lambda s: k(s.right(d))
def _loop(n):
    def f(start, stop, body, state):
        if start >= stop: return state,[]
        state, thisIteration = body(start)(state)
        state, laterIterations = f(start + 1, stop, body, state)
        return state, thisIteration + laterIterations
    def sequence(b,k,h):
        h,bodyBlocks = f(0,n,b,h)
        h,laterBlocks = k(h)
        return h,bodyBlocks+laterBlocks
    return lambda b: lambda k: lambda h: sequence(b,k,h)
def _simpleLoop(n):
    def f(start, body, k):
        if start >= n: return k
        return body(start)(f(start + 1, body, k))
    return lambda b: lambda k: f(0,b,k)
def _embed(body):
    def f(k):
        def g(hand):
            bodyHand, bodyActions = body(_empty_tower)(hand)
            # Record history if we are doing that
            if hand.history is not None:
                hand = TowerState(hand=hand.hand,
                                  orientation=hand.orientation,
                                  history=bodyHand.history)
            hand, laterActions = k(hand)
            return hand, bodyActions + laterActions
        return g
    return f
def _moveHand(n):
    return lambda k: lambda s: k(s.move(n))
def _reverseHand(k):
    return lambda s: k(s.reverse())
    
class TowerContinuation(object):
    def __init__(self, x, w, h):
        self.x = x
        self.w = w*2
        self.h = h*2
    def __call__(self, k):
        def f(hand):
            thisAction = [(self.x + hand.hand,self.w,self.h)]
            hand = hand.recordBlock(thisAction[0])
            hand, rest = k(hand)
            return hand, thisAction + rest
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


ttower = baseType("tower")
common_primitives = [
    Primitive("tower_loopM", arrow(tint, arrow(tint, ttower, ttower), ttower, ttower), _simpleLoop),
    Primitive("tower_embed", arrow(arrow(ttower,ttower), ttower, ttower), _embed),
] + [Primitive(name, arrow(ttower,ttower), TowerContinuation(0, w, h))
     for name, (w, h) in blocks.items()] + \
         [Primitive(str(j), tint, j) for j in range(1,9) ]
primitives = common_primitives + [
    Primitive("left", arrow(tint, ttower, ttower), _left),
    Primitive("right", arrow(tint, ttower, ttower), _right)
    ]

new_primitives = common_primitives + [
    Primitive("moveHand", arrow(tint, ttower, ttower), _moveHand),
    Primitive("reverseHand", arrow(ttower, ttower), _reverseHand)
    ]

def executeTower(p, timeout=None):
    try:
        return runWithTimeout(lambda : p.evaluate([])(_empty_tower)(TowerState())[1],
                              timeout=timeout)
    except RunWithTimeout: return None
    except: return None

def animateTower(exportPrefix, p):
    print(exportPrefix, p)
    from dreamcoder.domains.tower.tower_common import renderPlan
    state,actions = p.evaluate([])(_empty_tower)(TowerState(history=[]))
    print(actions)
    trajectory = state.history + [state]
    print(trajectory)
    print()

    assert tuple(z for z in trajectory if not isinstance(z, TowerState) ) == tuple(actions)        

    def hd(n):
        h = 0
        for state in trajectory[:n]:
            if isinstance(state, TowerState):
                h = state.hand
        return h
    animation = [renderPlan([b for b in trajectory[:n] if not isinstance(b, TowerState)],
                            pretty=True, Lego=True,
                            drawHand=hd(n),
                            masterPlan=actions,
                            randomSeed=hash(exportPrefix))
                 for n in range(0,len(trajectory) + 1)]
    import scipy.misc
    import random
    r = random.random()
    paths = []
    for n in range(len(animation)):
        paths.append(f"{exportPrefix}_{n}.png")
        scipy.misc.imsave(paths[-1], animation[n])
    os.system(f"convert -delay 10 -loop 0 {' '.join(paths)} {exportPrefix}.gif")
#    os.system(f"rm {' '.join(paths)}")
