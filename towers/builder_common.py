# Common procedures
# Kevin Ellis 2013, 6.868 Final Project

from panda3d.core import Vec3
from pandac.PandaModules import NodePath
from pandac.PandaModules import Point3
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletCylinderShape
from panda3d.bullet import BulletGhostNode
from panda3d.bullet import *

import math
import random
import cProfile
import os

ground_mass = 10000000.

floor_height = -3
stage_width = 10

# Constant block sizes
BLOCKLONGDIMENSION = 1.5
BLOCKSHORTDIMENSION = 0.25

# Timestep size
dt = 0.1

perturb_strength = 5.0
perturb_duration = 0.6

# Given a plan represented as a string, turn it in to a list of tuples
def parse_plan(plan):
    plan = eval(plan)
    newPlan = []
    for p in plan:
        if len(p) == 2:
            if p[1]:
                newPlan.append((p[0],1.5,0.25))
            else:
                newPlan.append((p[0],0.25,1.5))
        else:
            newPlan.append(p)
    return newPlan


# Returns the greatest vertical extent of a dynamic object
def get_construction_height(dynamic_objects):
    height = float("-inf") # Greatest height found so far
    for np in dynamic_objects:
        h   = np.getPos().z
        ang = np.getHpr().z
        # Panda doesn't use radians!
        ang = ang / 180.0 * math.pi
        ext = np.node().getShape(0).getHalfExtentsWithoutMargin()
        dx  = ext.x
        dz  = ext.z
        
        r = math.sqrt(dx*dx + dz*dz)
        dtheta = math.atan2(dz, dx)
            
        height = max(height, h + r * math.sin(ang + dtheta))
        height = max(height, h + r * math.sin(ang - dtheta))
        height = max(height, h + r * math.sin(math.pi + ang + dtheta))
        height = max(height, h + r * math.sin(math.pi + ang - dtheta))
    return height

# Sets up walls and a floor
def make_initial_world():
    world = BulletWorld()
    world.setGravity(Vec3(0, 0, -9.81))
    gnd = make_box(world, None, 0, floor_height, stage_width*5, 0.05, 'Ground',
                   mass = ground_mass,
                   friction = 5.0)
    if False:
        make_box(world, None, -stage_width/2, 3, 0.5, 6, 'LeftWall',
                 mass = 0.0)
        make_box(world, None, stage_width/2, 3, 0.5, 6, 'RightWall',
                 mass = 0.0)
    
    # Stop the ground from falling
    gnd.node().setGravity((0,0,0))
    gnd.node().setLinearFactor((1,0,0))
    
    return world, gnd


box_cache = []
def make_box(world, parent,
             x, z, dx, dz, name,
             mass = 0.1,
             friction = 0.5,
             r=None, g=None, b=None):
    global box_cache
    np = None
    for (dx_, dz_, np_) in box_cache:
        if dx_ == dx and dz_ == dz:
            np = np_
            box_cache.remove((dx_, dz_, np_))
            if parent != None:
                np.reparentTo(parent)
            break
    if np == None:
        node = BulletRigidBodyNode(name)
        node.setMass(mass)
        node.setLinearFactor((1,0,1))
        node.setAngularFactor((0,1,0))
        shape = BulletBoxShape(Vec3(dx, 0.5, dz))
        shape.setMargin(0.)
        node.addShape(shape)
        if parent:
            np = parent.attachNewNode(node)
        else:
            np = NodePath(node)
    
    np.setPosHprScale(x, 0, z, 0, 0, 0, 1, 1, 1)
    node = np.node()
    
    node.setLinearVelocity(Vec3(0,0,0))
    node.setAngularVelocity(Vec3(0,0,0))
        
    world.attachRigidBody(node)
    node.setActive(True)
    
    if r != None and b != None and g != None:
        np.composeColorScale(r,g,b,1.0)
        model = loader.loadModel("./box-round.egg")
        model.flattenLight()
        model.reparentTo(np)
        model.setScale(dx/0.5, 1, dz/0.5)
    
    return np

def remove_box(world, np):
    node = np.node()
    world.remove(node)
    
    global box_cache
    ext = node.getShape(0).getHalfExtentsWithoutMargin()
    dx  = ext.x
    dz  = ext.z
    box_cache.append((dx, dz, np))

def clear_boxes(world, boxes):
    for box in boxes:
        remove_box(world, box)

# If a block has fallen, then it returns None
def scene_stationary(world):
    bodies = world.getRigidBodies()
    
    for body in bodies:
        if body.getShapePos(0).z < floor_height - 1:
            return None
        v = body.getLinearVelocity()
        if v.x != 0 or v.z != 0:
            return False
        a = body.getAngularVelocity()
        if v.y != 0:
            return False
    return True

def run_until_stationary(world):
    for i in range(0,1000):
        world.doPhysics(dt)
    
    t_total = 0.0
    
    while True:
        stat = scene_stationary(world)
        if stat == None: return None
        if stat == True: return True
        world.doPhysics(dt)
        t_total += dt
        if t_total > 20.0: return None

def perturb_floor(world, gnd,
                  strength=perturb_strength):
    gnd.node().setActive(True)
    gnd.node().setLinearVelocity(Vec3(strength,0,0))
    
    for i in range(0, int(perturb_duration/dt)):
        gnd.node().setActive(True)
        world.doPhysics(dt)
        
    gnd.node().setLinearVelocity(Vec3(0,0,0))
    gnd.setPos(0, 0, floor_height)

    world.doPhysics(dt)

def impart_random_impulses(world, dynamic_objects,
                           strength=perturb_strength):
    for np in dynamic_objects:
        np.node().setActive(True)
        if random.random() > 0.5:
            np.node().setAngularVelocity(Vec3(0, strength, 0))
        else:
            np.node().setAngularVelocity(Vec3(0, -strength, 0))
        ang = random.random() * 2 * math.pi
        vx = strength * math.cos(ang)
        vy = strength * math.sin(ang)
        np.node().setLinearVelocity(Vec3(vx,0,vy))

def lowest_allowed_z(world, x, z, dx, dz):
    epsilon_z = 0.01
    
    while is_legal_action(world, x, z, dx, dz):
        z = z - epsilon_z
        if z < floor_height-1.0:
            return None
    
    return z + epsilon_z

# Can we legally place a block at x,z having extents dx,dz?
legality_cache = []
def is_legal_action(world, x, z, dx, dz):
    global legality_cache
    np = None
    for (dx_, dz_, np_) in legality_cache:
        if dx_ == dx and dz_ == dz:
            np = np_
            break
    if np == None:
        node = BulletRigidBodyNode("legality_test")
        shape = BulletBoxShape(Vec3(dx, 0.5, dz))
        node.addShape(shape)
        np = NodePath(node)
        legality_cache.append((dx,dz,np))
    np.setPos(x, 0, z)
    return world.contactTest(np.node()).getNumContacts() == 0

def run_plan(world, plan, render=None):
    boxes = []
    for plan_atom in plan:
        if len(plan_atom) == 4:
            (x, z, dx, dz) = plan_atom
        elif len(plan_atom) == 2:
            x, orientation = plan_atom
            if orientation:
                dx = BLOCKLONGDIMENSION
                dz = BLOCKSHORTDIMENSION
            else:
                dx = BLOCKSHORTDIMENSION
                dz = BLOCKLONGDIMENSION
            plan_atom = (x,dx,dz)

        if len(plan_atom) == 3:
            (x, dx, dz) = plan_atom
            # Hack: it's impossible to have a tower taller than 20.0
            z = lowest_allowed_z(world, x, 20.0, dx, dz)
            if z == None:
                clear_boxes(world, boxes)
                return None

        if render:
            new_box = make_box(world, render, x, z, dx, dz, 'planNode',
                               r=random.random(),
                               g=random.random(),
                               b=random.random())
        else:
            new_box = make_box(world, None, x, z, dx, dz, 'planNode')
        boxes.append(new_box)
        # Save positions/orientations of all of the boxes; if they change sufficiently, then the plan isn't stable
        positions = []
        for b in boxes:
            positions.append(b.getPos())
        # Simulate physics
        if run_until_stationary(world) == None:
            clear_boxes(world, boxes)
            return None
        newpos = []
        for b in boxes:
            newpos.append(b.getPos())
        # Check to see if anything moved too much
        for j in range(0,len(boxes)):
            b = boxes[j]
            p = positions[j]
            p_ = b.getPos()
            d = (p-p_).lengthSquared()
            if d > 1:
                clear_boxes(world, boxes)
                return None
    return boxes

def sample_stability(strength, world, boxes, saved, ht):
    timesStable = 0
    numSamples = 5
    
    for i in range(0,numSamples):
        impart_random_impulses(world, boxes, strength)
        if run_until_stationary(world) == None:
            pass
        else:
            h = get_construction_height(boxes)
            if h >= ht - 0.05:
                timesStable += 1
        
        for (p,h,box) in saved:
            box.setPos(p)
            box.setHpr(h)
    
    # Return percentage success rate
    return int(100.0*float(timesStable)/float(numSamples))


