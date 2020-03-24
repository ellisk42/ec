#missing imports

import os
#import time
from dreamcoder.program import Hole
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.utilities import *
from dreamcoder.zipper import *
from dreamcoder.grammar import NoCandidates
#from dreamcoder.type import Context
import time

import numpy as np

class SearchResult:
    def __init__(self, program, loss, time, evaluations):
        self.evaluations = evaluations
        self.program = program
        self.loss = loss
        self.time = time

class Solver():
    def __init__():
        pass

    def infer(self, g, tasks, likelihoodModel, _=None,
                          #verbose=False,
                          timeout=None,
                          elapsedTime=0.,
                          CPUs=1,
                          testing=False, #unused
                          evaluationTimeout=None,
                          # lowerBound=0.,
                          # upperBound=100.,
                          # budgetIncrement=1.0, 
                          maximumFrontiers=None):
        return frontiers, searchTimes, totalNumberOfPrograms


"""
TODO:
- [X] g.sampleFromSketch
- [X] hasHoles
"""

class SMC(Solver):
    def __init__(self, owner, _=None,
                 maximumLength=20,
                 initialParticles=8, exponentialGrowthFactor=2,
                 criticCoefficient=1.,
                 maxDepth=8,
                 holeProb=0.2):
        self.maximumLength = maximumLength
        self.initialParticles = initialParticles
        self.exponentialGrowthFactor = exponentialGrowthFactor
        self.criticCoefficient = criticCoefficient
        self.owner = owner
        self.maxDepth = maxDepth
        self.holeProb = holeProb

    def infer(self, g, tasks, likelihoodModel, _=None,
                              #verbose=False,
                              timeout=None,
                              elapsedTime=0.,
                              CPUs=1,
                              testing=False, #unused
                              evaluationTimeout=None,
                              # lowerBound=0.,
                              # upperBound=100.,
                              # budgetIncrement=1.0, 
                              maximumFrontiers=None): #IDK what this is...

        class Particle():
            def __init__(self, trajectory, zippers, frequency, finished=False):
                self.frequency = frequency
                self.trajectory = trajectory
                #self.graph = ProgramGraph(trajectory)
                self.zippers = zippers
                self.distance = None
                self.finished = finished
                self.reported = False
            def __str__(self):
                return f"Particle(frequency={self.frequency}, -logV={self.distance}, finished={self.finished}, sketch={self.trajectory}" #, graph=\n{self.graph.prettyPrint()}\n)"
            @property
            def immutableCode(self):
                return (self.graph, self.finished)    
            def __eq__(self,o):
                return self.trajectory == o.trajectory
            def __ne__(self,o): return not (self == o)
            def __hash__(self): return hash(self.trajectory)

        #START
        assert timeout is not None, \
            "enumerateForTasks: You must provide a timeout."

        request = tasks[0].request
        assert all(t.request == request for t in tasks), \
            "enumerateForTasks: Expected tasks to all have the same type"

        #assert len(tasks) == 1, "only should be one task"
        if not all(task == tasks[0] for task in tasks):
            print("WARNING, tasks are not all the same")
        task = tasks[0]

        self.maximumFrontiers = [maximumFrontiers[t] for t in tasks]
        # store all of the hits in a priority queue
        # we will never maintain maximumFrontier best solutions

        self.allHits = set()
        hits = [PQ() for _ in tasks]

        self.reportedSolutions = {t: [] for t in tasks}

        numberOfParticles = self.initialParticles #TODO
        allObjects = set()
        
        starting = time.time()
        # previousBudget = lowerBound
        # budget = lowerBound + budgetIncrement

        totalNumberOfPrograms = 0

        while time.time() - starting < timeout:
            # this line ensures particles start with a hole, wrapped in appropriate number of lambdas
            h = baseHoleOfType(request)
            zippers = findHoles(h, request)
            population = [Particle(h, zippers, numberOfParticles)]

            for generation in range(self.maximumLength):
                if time.time() - starting > timeout: break
                
                sampleFrequency = {} # map from (trajectory, finished) to frequency
                skToZippers = {} # map from sketch to zippers

                newObjects = set()
                for p in population:

                    for _ in range(p.frequency):

                        try:
                            newObject, newZippers = sampleSingleStep(g, p.trajectory,
                                                    request, holeZippers=p.zippers,
                                                    maximumDepth=self.maxDepth)
                        except NoCandidates:
                            print(f"NoCand error on particle: {p}")
                            break

                    # for newObject, newZippers in (sampleSingleStep(g, p.trajectory,
                    #                                 request, holeZippers=p.zippers,
                    #                                 maximumDepth=self.maxDepth)
                    #                                     for _ in range(p.frequency)):

                        #print("NO", newObject)
                        #print("NZ", newZippers)

                        if newObject is None:
                            assert False
                            #newKey = (p.trajectory, True)
                        elif newZippers == []: #TODO optimize
                            #print("hit finished branch")
                            newKey = (newObject, True)
                        else:
                            #print("hit unfin branch ")
                            #newKey = (tuple(list(p.trajectory) + [newObject]), False)
                            newKey = (newObject, False)


                        if newObject not in allObjects:
                            newObjects.add(newObject)
                            
                        sampleFrequency[newKey] = sampleFrequency.get(newKey, 0) + 1
                        
                        if newObject not in skToZippers:
                            skToZippers[newObject] = newZippers

                #print(sampleFrequency)

                for o in newObjects: allObjects.add(o)

                for p, f in sampleFrequency:
                    if f:
                        totalNumberOfPrograms = self._report(p, request, g, tasks,
                                                            likelihoodModel, hits, 
                                                            starting, elapsedTime, 
                                                            totalNumberOfPrograms) # TODO

                # Convert trajectories to particles
                samples = [Particle(t, skToZippers[t], frequency, finished=finished)
                           for (t, finished), frequency in sampleFrequency.items() ]
                
                # Computed value
                for p in samples:
                    # SHOULD I Resample with or without the finished ones? if not, then i lose particles along the way
                    #print("HIT THE COMPUTE VALUE LINE")
                    if p.finished:
                        p.distance = 0. if p.trajectory in self.allHits else 10^10
                    else:
                        p.distance = self.owner.valueHead.computeValue(p.trajectory, task) #memoize by registering tasks or something

                # Resample
                logWeights = [math.log(p.frequency) - p.distance*self.criticCoefficient 
                              for p in samples] # TODO
                ps = [ math.exp(lw - max(logWeights)) for lw in logWeights ]
                ps = np.array(ps)
                ps = ps/(np.sum(ps) + 1e-15)
                #TODO error
                try:
                    sampleFrequencies = np.random.multinomial(numberOfParticles, ps)
                except ValueError:
                    print("logweights")
                    print(logWeights)
                    print("probs")
                    print(ps)

                population = []

                for particle, frequency in sorted(zip(samples, sampleFrequencies),
                                                  key=lambda sf: sf[1]):

                    particle.frequency = frequency
                    # if frequency > 0.3*numberOfParticles:
                    #     print(particle)
                    #     print()
                    if frequency > 0 and not particle.finished:
                        particle.frequency = frequency
                        population.append(particle)
                        
                if len(population) == 0: break
                
            numberOfParticles *= self.exponentialGrowthFactor
            #print("Increased number of particles to", numberOfParticles)

        ## FINISH
        frontiers = {tasks[n]: Frontier([e for _, e in hits[n]],
                                        task=tasks[n])
                     for n in range(len(tasks))}
        searchTimes = {
            tasks[n]: None if len(hits[n]) == 0 else \
            min(t for t,_ in hits[n]) for n in range(len(tasks))}

        return frontiers, searchTimes, totalNumberOfPrograms, self.reportedSolutions

    def _report(self, p, request, g, tasks, likelihoodModel, hits, starting, elapsedTime, totalNumberOfPrograms):
        totalNumberOfPrograms += 1
        prior = g.logLikelihood(request, p) # TODO for speed can compute this at sample time

        for n in range(len(tasks)):
            #assert n == 0, "for now, just doing one task per thread seems reasonable"
            task = tasks[n]
            #Warning:changed to max's new likelihood model situation
            #likelihood = task.logLikelihood(p, evaluationTimeout)
            #if invalid(likelihood):
                #continue
            success, likelihood = likelihoodModel.score(p, task)
            if not success:
                continue

            self.allHits.add(p)

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

        return totalNumberOfPrograms
