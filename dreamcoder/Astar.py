

import os
#import time
import sys
from dreamcoder.program import Hole
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.utilities import *
from dreamcoder.zipper import *
from dreamcoder.grammar import NoCandidates
#from dreamcoder.type import Context
import time
import numpy as np

from dreamcoder.SMC import SearchResult, Solver

from queue import PriorityQueue
import signal
"""
TODO
- [X] node rep: might need object
- [X] queue syntax and comparison op
- [X] reporting
- [X] finishing & returning
- [ ] record node expansions or whatever (both ways), search times
- [X] enumSingleStep 2x
- [X] all start-up stuff

"""
class InferenceTimeout(Exception):
    pass

class Astar(Solver):

    def __init__(self, owner, _=None,
                 maximumLength=20,
                 #initialParticles=8, exponentialGrowthFactor=2,
                 criticCoefficient=1.,
                 maxDepth=16,
                 holeProb=0.2,
                 reportNodeExpansions=True):

        #which type of reporting: full programs or sketches:
        self.reportNodeExpansions = reportNodeExpansions 
        self.maximumLength = maximumLength
        #self.initialParticles = initialParticles
        #self.exponentialGrowthFactor = exponentialGrowthFactor
        self.criticCoefficient = criticCoefficient
        self.owner = owner
        self.maxDepth = maxDepth
        self.holeProb = holeProb

    def _getNextNodes(self, task, node, g, request):
        totalCost, policyCost, sketch, zippers = node
        #print("TOTAL COSTS", totalCost)
        if self.canonicalOrdering:
            zippers = [zippers[0]] #only use the first zipper if cannonicalOrdering
        for zipper in zippers:
            try:
                for stepCost, newZippers, newSketch in self.owner.policyHead.enumSingleStep(task, g, sketch, request, 
                                                                    holeZipper=zipper,
                                                                    maximumDepth=self.maxDepth):
                    yield policyCost + stepCost, newZippers, newSketch
            except AssertionError:
                #print("SKKKK", sketch)
                return

    def infer(self, g, tasks, likelihoodModel, _=None,
                              #verbose=False,
                              timeout=None,
                              elapsedTime=0.,
                              CPUs=1,
                              testing=False, #unused
                              evaluationTimeout=None, 
                              maximumFrontiers=None): #IDK what this is...
        
        if hasattr(self.owner.policyHead, 'canonicalOrdering') and self.owner.policyHead.canonicalOrdering:
            self.canonicalOrdering = True
        else: self.canonicalOrdering = False

            
        sys.setrecursionlimit(5000)
                #START
        assert timeout is not None, \
            "enumerateForTasks: You must provide a timeout."

        request = tasks[0].request
        assert all(t.request == request for t in tasks), \
            "enumerateForTasks: Expected tasks to all have the same type"
        assert len(tasks) == 1, "only should be one task"
        # if not all(task == tasks[0] for task in tasks):
        #     print("WARNING, tasks are not all the same")
        task = tasks[0]
        self.maximumFrontiers = [maximumFrontiers[t] for t in tasks]
        # store all of the hits in a priority queue
        self.fullPrograms = set()
        hits = [PQ() for _ in tasks]

        self.reportedSolutions = {t: [] for t in tasks}

        allObjects = set()
        
        starting = time.time()
        totalNumberOfPrograms = 0


        q = PQMaxSize(maxSize=1000000)
        #base node
        h = baseHoleOfType(request)
        zippers = findHoles(h, request)
        q.push(0., (0., 0., h, zippers))


        try:
            def timeoutCallBack(_1, _2): raise InferenceTimeout()
            # signal.signal(signal.SIGVTALRM, timeoutCallBack)
            # signal.setitimer(signal.ITIMER_VIRTUAL, timeout)  

            #signal.signal(signal.SIGALRM, timeoutCallBack)
            #signal.setitimer(signal.ITIMER_REAL, timeout) ##TODO TEMP DISABLED


            while time.time() - starting < timeout:

                node = q.popMaximum() #TODO
                #print(">>>>>>node", node)
                #print("queue size", len(q))

                nNei = 0
                for policyCost, zippers, neighbor in self._getNextNodes(task, node, g, request):
                    # print(policyCost, neighbor)
                    # import pdb; pdb.set_trace()
                    nNei += 1
                    if (neighbor) in allObjects:
                        continue
                    allObjects.add(neighbor)
                    if self.reportNodeExpansions: totalNumberOfPrograms += 1

                    if not zippers:
                        success, totalNumberOfPrograms = self._report(neighbor, policyCost, 
                                                                    request, g, tasks, 
                                                                    likelihoodModel,
                                                                    hits, 
                                                                    starting, 
                                                                    elapsedTime, 
                                                                    totalNumberOfPrograms) #TODO
                        if success: return self._finish(tasks,
                                                        hits, 
                                                        totalNumberOfPrograms)
                        else: continue

                    valueCost = self.owner.valueHead.computeValue(neighbor, task) #TODO 
                    totalCost = policyCost - self.criticCoefficient * valueCost #TODO normalize and scale
                    #print("neighbor:", neighbor)
                    #print("policyCost", policyCost)
                    #print("valueCost", valueCost)
                    #print("totalCost", totalCost)
                    #import pdb; pdb.set_trace()
                    newNode = (totalCost, policyCost, neighbor, zippers)
                    q.push(totalCost, newNode)


                    if time.time() - starting > timeout:
                        break
                #print('\t num neighbors', nNei)

        except InferenceTimeout:
            #print("Timed out while evaluating, timeout", timeout)
            #print("time elapsed")
            #print(time.time() - starting)
            pass
        finally:
            # signal.signal(signal.SIGVTALRM, lambda *_: None)
            # signal.setitimer(signal.ITIMER_VIRTUAL, 0)
            try:
                signal.signal(signal.SIGALRM, lambda *_: None)
                signal.setitimer(signal.ITIMER_REAL, 0)
            except InferenceTimeout:
                pass

        return self._finish(tasks, hits, totalNumberOfPrograms)

    def _report(self, p, prior, request, g, tasks, 
                likelihoodModel,
                hits, 
                starting, 
                elapsedTime, 
                totalNumberOfPrograms):
        if not self.reportNodeExpansions:
            totalNumberOfPrograms += 1

        if p in self.fullPrograms:
            return totalNumberOfPrograms
        else:
            self.fullPrograms.add(p)

        # prior = g.logLikelihood(request, p) # TODO for speed can compute this at sample time
        # prior
        for n in range(len(tasks)):
            #assert n == 0, "for now, just doing one task per thread seems reasonable"
            task = tasks[n]

            success, likelihood = likelihoodModel.score(p, task)
            if not success:
                continue

            dt = time.time() - starting + elapsedTime
            priority = -(likelihood + prior)
            hits[n].push(priority,
                         (dt, FrontierEntry(program=p,
                                            logLikelihood=likelihood,
                                            logPrior=prior)))

            if len(hits[n]) > self.maximumFrontiers[n]:
                hits[n].popMaximum()
            self.reportedSolutions[task].append(SearchResult(p, -priority, dt,
                                                       totalNumberOfPrograms))

        return success, totalNumberOfPrograms


    def _finish(self,
                tasks,
                hits, 
                totalNumberOfPrograms):        
        frontiers = {tasks[n]: Frontier([e for _, e in hits[n]],
                                        task=tasks[n])
                     for n in range(len(tasks))}
        searchTimes = {
            tasks[n]: None if len(hits[n]) == 0 else \
            min(t for t,_ in hits[n]) for n in range(len(tasks))}

        return frontiers, searchTimes, totalNumberOfPrograms, self.reportedSolutions

