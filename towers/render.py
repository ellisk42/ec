#!/usr/bin/python

# This allows one to visualize the construction of towers
# Kevin Ellis 2013, 6.868 Final Project


from direct.task import Task
from panda3d.core import Vec3
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletCylinderShape
from panda3d.bullet import BulletGhostNode
from panda3d.bullet import *
from direct.showbase.ShowBase import ShowBase
from pandac.PandaModules import NodePath
import pandac.PandaModules as pm
from panda3d.core import *
import os
import sys
import pdb
import time
import math
import random

from builder_common import *


class BuildBase(ShowBase):
    def __init__(self, plan, testShove = None):
        ShowBase.__init__(self)
        self.disableMouse()
        self.win.setClearColor((139.0/255.0, 125.0/255.0, 107.0/255.0, 1.0))
        self.cam.setPos(4, -20, 0)
        self.cam.lookAt(0, 0, 0)
        
        self.accept('escape', sys.exit)
        
        self.dynamic_objects = []
        
        # Create the plane in which the block world will live
        # This is used for mouse control;
        # we check where the mouse intersects the plane
        self.background_plane = pm.Plane(Vec3(0,1,0),pm.Point3(0,0,0))
        
        # Set up Bullet
        self.world, self.gnd_np = make_initial_world()
        
        self.imparted_impulse = False
        self.perturbing_floor = False
        
        # Add a decorative floor
        self.add_box(0.0, floor_height, stage_width*4, 0.1, 0.9, 0.9, 0.9,
                     "Floor!", dy=1.0)
        
        # Uncomment this line to turn on Bullet debug rendering
        # self.enable_debug_render()
        
        self.make_lighting()
        
        # Plan is a list of actions to automatically take
        self.savedPlan = list(plan)
        self.plan = plan
        
        self.testShove = testShove
        
        self.taskMgr.add(self.run_physics, "run_physics")
        self.taskMgr.add(self.run_plan, 'run_plan')
        self.taskMgr.add(self.update_freeze, 'update_freeze')
        self.accept('mouse2',self.perturb_floor)
        
        # When the world is frozen, all dynamic_objects are made static
        # They cannot be unfrozen until unfreeze_time has elapsed,
        #   AND movement has ceased
        self.freeze_world = False
        self.unfreeze_time = time.time() + 1
        
    
    def update_freeze(self, task):
        if not self.freeze_world: # Check if we should freeze everything
            if time.time() > self.unfreeze_time:
                if scene_stationary(self.world):
                    self.freeze_world = True
                    if len(self.plan) == 0 and self.testShove != None and self.imparted_impulse == False:
                        impart_random_impulses(self.world,
                                               self.dynamic_objects,
                                               self.testShove)
                        #self.testShove = None
                        self.freeze_world = False
                        self.unfreeze_time = time.time() + 1.0
                        self.imparted_impulse = True
                        self.taskMgr.doMethodLater(1.5, self.reset_plan, 'reset_plan')
        return task.cont
    
    def reset_plan(self, task):
        if scene_stationary(self.world):
            self.plan = list(self.savedPlan)
            clear_boxes(self.world, self.dynamic_objects)
            for box in self.dynamic_objects:
                box.detachNode()
            self.dynamic_objects = []
            self.imparted_impulse = False
        else:
            return task.cont
    
    def make_lighting(self):
        # Make a point light
        plight = pm.PointLight('plight')
        plight.setColor(pm.VBase4(1.0, 1.0, 1.0, 1))
        plnp = self.render.attachNewNode(plight)
        plnp.setPos(0, -5, 2)
        self.render.setLight(plnp)
    
    def run_physics(self, task):
        dt = globalClock.getDt()
        if not self.freeze_world or self.perturbing_floor:
            self.world.doPhysics(dt)
        if self.perturbing_floor: self.gnd_np.node().setActive(True)
        return task.cont
    
    def run_plan(self, task):
        if len(self.plan) > 0 and self.freeze_world:
            # Place the next block in the plan,
            # and remove it from the plan; use random colors
            planlet = self.plan[0]
            if len(planlet) == 3:
                (x, dx, dz) = planlet
            elif len(planlet) == 2:
                (x, sidep) = planlet
                if sidep:
                    dx = BLOCKLONGDIMENSION
                    dz = BLOCKSHORTDIMENSION
                else:
                    dx = BLOCKSHORTDIMENSION
                    dz = BLOCKLONGDIMENSION
            z = lowest_allowed_z(self.world, x, 20.0, dx, dz)
            if z == None:
                pdb()
            else:
                #z += 1.2
                pass
            self.plan = self.plan[1:]
            np = make_box(self.world,
                          self.render,
                          x, z, dx, dz, 'planNode',
                          r=random.random(),
                          g=random.random(),
                          b=random.random())
            self.dynamic_objects.append(np)
            self.freeze_world = False
            self.unfreeze_time = time.time() + 0.5
        return task.cont
    
    def perturb_floor(self, strength=10.0):
        print 'trying perturb'
        if not self.perturbing_floor:
            print 'got perturb'
            self.gnd_np.node().setActive(True)
            self.perturbing_floor = True
            
            # Allow the floor to move in the x direction
            self.gnd_np.node().setLinearFactor((1,0,0))
            
            # Impart impulse
            self.gnd_np.node().setLinearVelocity(Vec3(strength,0,0))
            
            self.freeze_world = False
            self.unfreeze_time = time.time() + 2.0
            
            # Schedule task to cancel impulse
            self.taskMgr.doMethodLater(perturb_duration, self.reset_floor, 'reset_floor')
    
    def reset_floor(self, task):
        print "reseting floor"
        print self.gnd_np.getPos()
        print self.gnd_np.node().getLinearVelocity()
        self.gnd_np.node().setLinearFactor((0,0,0))
        self.gnd_np.node().setLinearVelocity((0, 0, 0))
        self.gnd_np.setPos(0, 0, floor_height)
        self.perturbing_floor = False
    
    # Adds a decorative box
    def add_box(self, x, z, dx, dz, r, g, b, name,
                transparent=False, dy=0.5):
        node = PandaNode(name)
        np = self.render.attachNewNode(node)
        np.setPosHprScale(x, 0, z, 0, 0, 0, 1, 1, 1)
        if transparent:
            np.setTransparency(TransparencyAttrib.MAlpha)
            np.setAlphaScale(0.5)
        
        self.beautify_np(np, dx, dz, r, g, b, dy=dy)
        return np
    
    def beautify_np(self, np, dx, dz, r, g, b, dy=0.5):
        np.reparentTo(self.render)
        model = loader.loadModel("box-round.egg")
        np.setColor(r,g,b,1.0)
        model.setColor(r,g,b,1.0)
        model.flattenLight()
        model.reparentTo(np)
        model.setScale(dx/0.5, dy/0.5, dz/0.5)
        
    def enable_debug_render(self):
        debugNode = BulletDebugNode('Debug')
        debugNode.showWireframe(True)
        debugNode.showConstraints(True)
        debugNode.showBoundingBoxes(False)
        debugNode.showNormals(False)
        debugNP = self.render.attachNewNode(debugNode)
        debugNP.show()
        self.world.setDebugNode(debugNP.node())

def main():
    random.seed()
    if len(sys.argv) < 2:
        base = BuildBase([])
        base.run()
    else:
        plan = eval(sys.argv[1])
        if len(sys.argv) == 3:
            base = BuildBase(plan, float(sys.argv[2]))
            base.run()
        else:
            base = BuildBase(plan, None)
            base.run()

main()
