#!/usr/bin/python

# Kevin Ellis 2013, 6.868 Final Project

from builder_common import *

import sys
import random

def handle(request):
   data = request.strip().split('|')
   plan = parse_plan(data[0])
   perturbations = map(float, data[1][1:-1].split(','))
   
   reply = []
   
   # Build it to see the height of the tower
   world, gnd = make_initial_world()
   boxes = run_plan(world, plan)
   if boxes == None:
      print "FAIL"
      sys.exit(0)
   
   # Save the locations of each of the blocks
   saved = []
   for box in boxes:
      p = box.getPos()
      h = box.getHpr()
      saved.append((p,h,box))
   
   # Save the height of the tower
   towerHeight = get_construction_height(boxes)
   reply.append(towerHeight)
   
   for perturbation in perturbations:
      percentStable = sample_stability(perturbation, world, boxes, saved, towerHeight)
      reply.append(percentStable)
      if percentStable < 1: break # Don't try stronger perturbations if this one fails
   
   print str(reply)

random.seed()
handle(sys.argv[1])
