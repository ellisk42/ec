from builder_common import *

import random

def towerLength(plan):
    return max( z + (BLOCKLONGDIMENSION if orientation else BLOCKSHORTDIMENSION)
                for z,orientation in plan ) - \
           min( z - (BLOCKLONGDIMENSION if orientation else BLOCKSHORTDIMENSION)
                for z,orientation in plan )

class TowerSimulationResult(object):
    def __init__(self, _ = None, height = None, stability = None, plan = None):
        self.height = height
        self.stability = stability

        # Calculate the width of the tower
        self.width = towerLength(plan)

    def __str__(self):
        return "TowerSimulationResult(height = %f, width = %f, stability = %s)"%(self.height,
                                                                                 self.width,
                                                                                 self.stability)

SIMULATIONCASH = {}
def simulateTower(plan, perturbations):
    global SIMULATIONCASH
    cashKey = (tuple(plan),tuple(perturbations))
    if cashKey in SIMULATIONCASH: return SIMULATIONCASH[cashKey]
    
    # Build it to see the height of the tower
    world, gnd = make_initial_world()
    boxes = run_plan(world, plan)
    if boxes == None:
        return None
        # print "FAIL"
        # import sys
        # sys.exit(0)
   
    # Save the locations of each of the blocks
    saved = []
    for box in boxes:
        p = box.getPos()
        h = box.getHpr()
        saved.append((p,h,box))
   
    # Save the height of the tower
    towerHeight = get_construction_height(boxes)

    stabilities = []
    for perturbation in sorted(perturbations):
        percentStable = sample_stability(perturbation, world, boxes, saved, towerHeight)
        stabilities.append(percentStable)
        if percentStable < 1: # Don't try stronger perturbations if this one fails
            stabilities += [0]*(len(perturbations) - len(stabilities))
            break
    result = TowerSimulationResult(stability = stabilities, height = towerHeight, plan = plan)
    SIMULATIONCASH[cashKey] = result
    return result
