import Box2D  # The main library
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)

from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody)

import random
import math

class TowerWorld(object):
    def __init__(self):
        self.world = world(gravity=(0, -10), doSleep=True)
        self.ground_body = self.world.CreateStaticBody(
            position=(0, 0),
            shapes=polygonShape(box=(50, 1)),
            userData = (255, 255, 255, 255)
        )

        self.blocks = []

        self.H = 3.
        self.W = 0.5
        self.dt = 1./60
        self.locationNoise = 0.13

    def placeBlock(self, x, orientation):
        x += 11

        x += random.random()*self.locationNoise - self.locationNoise/2
        
        if orientation: # horizontal
            safetyMargin = self.W/2
        else: #vertical
            safetyMargin = self.H/2

        dx = self.H if orientation else self.W
        dx = dx/2
        #print "About to place a block at X = ",x,"dx = ",dx
        minimumX = x - dx
        maximumX = x + dx
        conflicts = []
        for b in self.blocks:
            xs = [ (b.transform * v)[0] for v in b.fixtures[0].shape.vertices ]
            #print "existing block has",xs
            
            if any( xp >= minimumX and xp <= maximumX for xp in xs ):
                #print "CONFLICT"
                conflicts += [ (b.transform * v)[1] for v in b.fixtures[0].shape.vertices ]
            #print 

        y = max(conflicts + [0.5]) + safetyMargin + 0.1
        #print "Decided to place the block at Y = ",y
        #print "safety margin",safetyMargin
        
        body = self.world.CreateDynamicBody(position=(x, y),
                                            angle=0 if not orientation else math.pi/2,
                                            userData = tuple(random.random()*128+127 for _ in range(4) ))
        box = body.CreatePolygonFixture(box=(self.W/2, self.H/2), density=1, friction=0.3)
        self.blocks.append(body)

        # print "Done placing block."
        # print
        # print
        # print 

    def step(self, dt):
        self.world.Step(dt, 10, 10)

    def unmoving(self):
        return all( abs(b.linearVelocity[0]) < 0.05 and abs(b.linearVelocity[1]) < 0.05
                    for b in self.blocks ) 

    def height(self):
        if self.blocks == []: return 0
        return max( (b.transform * v)[1]
                 for b in self.blocks
                 for v in b.fixtures[0].shape.vertices )

    def length(self):
        if self.blocks == []: return 0
        xs = [ (b.transform * v)[0]
               for b in self.blocks
               for v in b.fixtures[0].shape.vertices ]
        return max(xs) - min(xs)

    def impartImpulses(self, p):
        for b in self.blocks:
            b.ApplyLinearImpulse([random.random()*p - p/2,
                                  random.random()*p],
                                 b.worldCenter,
                                 True)
            b.ApplyAngularImpulse(random.random()*p - p/2,
                          True)

    def stepUntilStable(self):
        for _ in range(10): self.step(self.dt)
        for _ in range(100000):
            self.step(self.dt)
            if self.unmoving(): break
        
    def executePlan(self, plan):
        initialHeight = float('-inf')
        badPlan = False
        for p in plan:
            self.placeBlock(*p)
            self.stepUntilStable()
            newHeight = self.height()
            if newHeight < initialHeight - 0.1: badPlan = True
            initialHeight = newHeight
        return not badPlan
    
    def clearWorld(self):
        for b in self.blocks:
            self.world.DestroyBody(b)
        self.blocks = []

    def sampleStability(self, plan, perturbation, N = 5):
        hs = []
        wasStable = []
        for _ in range(N):
            planSucceeds = self.executePlan(plan)
            if planSucceeds:
                initialHeight = self.height()
                hs.append(initialHeight)
                self.impartImpulses(perturbation)
                self.stepUntilStable()
                wasStable.append(self.height() > initialHeight - 0.1)
            else:
                hs.append(0.)
                wasStable.append(False)

            # reset the world
            self.clearWorld()
        h = sum(hs)/N
        return h, wasStable
    
            

