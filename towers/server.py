# This listens for plans on port 1540 and then runs them,
# returning the result

# Kevin Ellis 2013, 6.868 Final Project

from builder_common import *

import SocketServer
import sys
import os

world, gnd = make_initial_world()

class BuildHandler(SocketServer.BaseRequestHandler):
    def handle(self):
        data = self.request.recv(2048).strip().split('|')
        plan = parse_plan(data[0])
        perturbations = map(float, data[1][1:-1].split(','))
        
        print "Running plan:", plan
        print "With perturbations:", perturbations
        
        reply = []
        
        for perturbation in perturbations:
            boxes = run_plan(world, plan)
            if boxes == None:
                reply.append([])
            else:
                if perturbation < 0.01:
                    new_reply = []
                    for box in boxes:
                        x   = box.getPos().x
                        z   = box.getPos().z
                        ang = box.getHpr().z
                        new_reply.append((x,z,ang))
                    reply.append(new_reply)
                else:
                    samples = sample_stability(perturbation, world, boxes)
                    reply.append(map(lambda x: (x, 0.0, 0.0), samples))
                clear_boxes(world, boxes)
        
        self.request.send(str(reply))
        print str(reply)
        self.request.close()

class BuildServer(SocketServer.TCPServer):
    allow_reuse_address = True
    
    def __init__(self, server_address, RequestHandlerClass):
        SocketServer.TCPServer.__init__(self, server_address, RequestHandlerClass)


server = BuildServer(("localhost", 1540), BuildHandler)
print "Listening on port 1540..."
server.serve_forever()
