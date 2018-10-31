import random
import math
from utilities import *


def simulateWithoutPhysics(plan):
    def overlap(b1,
                b2):
        (x,w,h) = b1
        (x_,y_,w_,h_) = b2
        x1 = x - w/2
        x2 = x + w/2
        x1_ = x_ - w_/2
        x2_ = x_ + w_/2
        if x1_ > x2 or x1 > x2_: return None
        return y_ + h_/2 + h/2
    def lowestPossibleHeight(b): return b[2]/2
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
    return list(sorted(w))

def centerTower(t,hand=None):
    if len(t) == 0:
        if hand is None:
            return t
        else:
            return t, hand
    x1 = max(x for x, _, _ in t)
    x0 = min(x for x, _, _ in t)
    c = (x1 - x0) / 2 + x0
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


def fastRendererPlan(plan, resolution=256, window=30, floorHeight=10,
                     borderSize=1, bodyColor=(0.,1.,1.),
                     borderColor=(1.,0.,0.),
                     pretty=False, Lego=False,
                     drawHand=None):
    import numpy as np
    if Lego: assert pretty

    if drawHand is not None:
        plan, drawHand = centerTower(plan, drawHand)
        drawHand = drawHand/10.
    else:
        plan = centerTower(plan)
    
    
    world = simulateWithoutPhysics(plan)
    world = [ [float(zz)/10. for zz in wb ]
              for wb in world ]
    a = np.zeros((resolution, resolution, 3))

    def transform(x,y):
        y = resolution - y*resolution/float(window)
        x = resolution/2 + x*resolution/float(window)
        return int(x + 0.5),int(y + 0.5)
    def clip(p):
        if p < 0: return 0
        if p >= resolution: return resolution - 1
        return int(p + 0.5)

    def _color():
        return random.random()*0.7 + 0.3
    def color():
        return (_color(),_color(),_color())
    def clear(x,y):
        for xp,yp,wp,hp in world:
            if x < xp + wp/2. and \
               x > xp - wp/2. and \
               y < yp + hp/2. and \
               y > yp - hp/2.:
                return False
        return True
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
    def bump(x,y,c):
        x,y = transform(x,y)
        y -= floorHeight
        y1 = y
        y2 = y - 1
        x1 = x - 1
        x2 = x + 1
        a[clip(y2) : clip(y1),
          clip(x1) : clip(x2),
          :] = c

        
    
    for x,y,w,h in world:
        w = int(w + 0.5)
        h = int(h + 0.5)
        x1,y1 = x - w/2., y - h/2.                          
        x2,y2 = x + w/2., y + h/2.

        if pretty:
            thisColor = color()
            rectangle(x1,x2,y1,y2,
                      thisColor)
            if Lego:
                bumps = int(w + 0.5)*2
                for nb in range(bumps):
                    nx = x - w/2. + 0.25 + nb*0.5
                    ny = y + h/2. + 0.0001
                    if clear(nx,ny):
                        bump(nx,ny,thisColor)
        else:
            rectangle(x1,x2,y1,y2,
                      borderColor, bodyColor)
        
        
    a[resolution - floorHeight:,:,:] = 1.

    if drawHand is not None:
        dh = 0.25
        rectangle(drawHand - dh,
                  drawHand + dh,
                  -99999, 99999,
                  (0,1,0))

    return a
                
        
    

def uglyTowerRender(plan, drawHand=None):
    import numpy as np

    if drawHand is not None:
        plan, drawHand = centerTower(plan, drawHand)
        drawHand = drawHand/10.
    else:
        plan = centerTower(plan)

    world = simulateWithoutPhysics(plan)
    world = [ [float(zz)/10. for zz in wb ]
              for wb in world ]

    a = np.zeros((resolution, resolution, 3))
    eprint(world)


