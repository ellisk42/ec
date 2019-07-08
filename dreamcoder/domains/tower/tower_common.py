import random
import math
from dreamcoder.utilities import *

def simulateWithoutPhysics(plan,ordered=True):
    def overlap(b1,
                b2):
        (x,w,h) = b1
        (x_,y_,w_,h_) = b2
        x1 = x - w/2
        x2 = x + w/2
        x1_ = x_ - w_/2
        x2_ = x_ + w_/2
        if x1_ >= x2 or x1 >= x2_: return None
        assert h%2 == 0 and h_%2 == 0
        return y_ + h_//2 + h//2
    def lowestPossibleHeight(b):
        h = b[2]
        assert h%2 == 0
        return int(h/2)
    def placeAtHeight(b,y):
        (x,w,h) = b
        return (x,y,w,h)
    def placeBlock(world, block):
        lowest = max([lowestPossibleHeight(block)] + \
                     [overlap(block,other)
                      for other in world
                      if overlap(block,other) is not None])
        world.append(placeAtHeight(block, lowest))

    w = []
    for p in plan: placeBlock(w,p)
    if ordered: w = list(sorted(w))
    return w

def centerTower(t,hand=None, masterPlan=None):

    if len(t) == 0:
        if hand is None:
            return t
        else:
            return t, hand
    def getCenter(t):
        x1 = max(x for x, _, _ in t)
        x0 = min(x for x, _, _ in t)
        c = int((x1 - x0) / 2.0) + x0
        return c
    c = getCenter(masterPlan or t)
    t = [(x - c, w, h) for x, w, h in t]
    if hand is None:
        return t
    else:
        return t, hand - c

def towerLength(t):
    if len(t) == 0: return 0
    x1 = max(x for x, _, _ in t)
    x0 = min(x for x, _, _ in t)
    return x1 - x0

def towerHeight(t):
    y1 = max(y + h/2 for _, y, _, h in t )
    y0 = min(y - h/2 for _, y, _, h in t )
    return y1 - y0



def renderPlan(plan, resolution=256, window=64, floorHeight=2, borderSize=1, bodyColor=(0.,1.,1.),
               borderColor=(1.,0.,0.),
               truncate=None, randomSeed=None,
               masterPlan=None,
               pretty=False, Lego=False,
               drawHand=None):
    import numpy as np

    if Lego: assert pretty

    if drawHand is not None and drawHand is not False:
        plan, drawHand = centerTower(plan, drawHand,
                                     masterPlan=masterPlan)
    else:
        plan = centerTower(plan,masterPlan=masterPlan)

    world = simulateWithoutPhysics(plan,
                                   ordered=randomSeed is None)
    if truncate is not None: world = world[:truncate]
    a = np.zeros((resolution, resolution, 3))

    def transform(x,y):
        y = resolution - y*resolution/float(window)
        x = resolution/2 + x*resolution/float(window)
        return int(x + 0.5),int(y + 0.5)
    def clip(p):
        if p < 0: return 0
        if p >= resolution: return resolution - 1
        return int(p + 0.5)
    def clear(x,y):
        for xp,yp,wp,hp in world:
            if x < xp + wp/2. and \
               x > xp - wp/2. and \
               y < yp + hp/2. and \
               y > yp - hp/2.:
                return False
        return True
    def bump(x,y,c):
        size = 0.5*resolution/window
        x,y = transform(x,y)
        y -= floorHeight
        y1 = y
        y2 = y - size
        x1 = x - size/2
        x2 = x + size/2
        a[clip(y2) : clip(y1),
          clip(x1) : clip(x2),
          :] = c


    if randomSeed is not None:
        randomNumbers = random.Random(randomSeed)
    def _color():
        if randomSeed is None:
            return random.random()*0.7 + 0.3
        else:
            return randomNumbers.random()*0.7 + 0.3
    def color():
        return (_color(),_color(),_color())
    
    def rectangle(x1,x2,y1,y2,c,cp=None):
        x1,y1 = transform(x1,y1)
        x2,y2 = transform(x2,y2)
        y1 -= floorHeight
        y2 -= floorHeight
        a[clip(y2) : clip(y1),
          clip(x1) : clip(x2),
          :] = c
        if cp is not None:
            a[clip(y2 + borderSize) : clip(y1 - borderSize),
              clip(x1 + borderSize) : clip(x2 - borderSize),
              :] = cp

    for x,y,w,h in world:
        x1,y1 = x - w/2., y - h/2.                          
        x2,y2 = x + w/2., y + h/2.
        if pretty:
            thisColor = color()
            rectangle(x1,x2,y1,y2,
                      thisColor)
            if Lego:
                bumps = w
                for nb in range(bumps):
                    nx = x - w/2. + 0.5 + nb
                    ny = y + h/2. + 0.00001
                    if clear(nx,ny):
                        bump(nx,ny,thisColor)
        else:
            rectangle(x1,x2,y1,y2,
                      borderColor, bodyColor)
    
    a[resolution - floorHeight:,:,:] = 1.
    if drawHand is not None:
        if not Lego:
            dh = 0.25
            rectangle(drawHand - dh,
                      drawHand + dh,
                      -99999, 99999,
                      (0,1,0))
        else:
            rectangle(drawHand - 1,drawHand + 1,
                      43,45,(1,1,1))

    return a


