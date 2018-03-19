from ec import explorationCompression, commandlineArguments, Program
from grammar import Grammar
from arithmeticPrimitives import real_addition, real_multiplication, real_power, real_subtraction, real, f0, f1
from task import DifferentiableTask, squaredErrorLoss, l1loss, Task
from type import tint, arrow
from utilities import *
from program import *
from recognition import *

tvector = baseType("vector")
treal = baseType("real")
tpositive = baseType("positive")

def makeTrainingData(request, law,
                     # Number of examples
                     N = 10,
                     # Vector dimensionality
                     D = 2,
                     # Maximum absolute value of a random number
                     S = 1):
    from random import random, randint

    def sampleArgument(a,listLength):
        if a.name == "real": return random()*S*2 - S
        elif a.name == "positive": return random()*S
        elif a.name == "vector":
            return [random() for _ in xrange(D) ]
        elif a.name == "list":
            return [sampleArgument(a.arguments[0], listLength) for _ in xrange(listLength) ]
        else:
            assert False, "unknown argument tp %s"%a
        
    
    arguments = request.functionArguments()
    e = []
    for _ in xrange(N):
        # Length of any requested lists
        l = randint(1,4)

        xs = tuple( sampleArgument(a,l) for a in arguments )
        y = law(*xs)
        e.append((xs,y))

    return e

def makeTask(name, request, law,
             # Number of examples
             N = 10,
             # Vector dimensionality
             D = 2,
             # Maximum absolute value of a random number
             S = 1):
    e = makeTrainingData(request, law,
                         N=N,D=D,S=S)
    def genericType(t):
        if t.name == "real": return treal
        elif t.name == "positive": return treal
        elif t.name == "vector": return tlist(treal)
        elif t.name == "list": return tlist(genericType(t.arguments[0]))
        elif t.isArrow():
            return arrow(genericType(t.arguments[0]),
                         genericType(t.arguments[1]))
        else:
            assert False, "could not make type generic: %s"%t

    return DifferentiableTask(name, genericType(request), e,
                              BIC = 0.,
                              likelihoodThreshold=-0.1,
                              loss = squaredErrorLoss)
        
def norm(v):
    return sum(x*x for x in v)**0.5
def scaleVector(a,v):
    return [a*x for x in v]

tasks = [
    # makeTask("Newtonian gravitation",
    #          arrow(tpositive, tpositive, tvector, tvector),
    #          lambda m1,m2,r: scaleVector(-0.1 * m1 * m2 / (norm(r)**2), r)),
    makeTask("Hook's law",
             arrow(tpositive,tpositive),
             lambda x: -2.*x*x,
             N = 20,
             S = 5),
    makeTask("Ohm's law",
             arrow(tpositive,tpositive,tpositive),
             lambda r,i: r*i,
             N = 20,
             S = 5)
    ]

if __name__ == "__main__":
    baseGrammar = Grammar.uniform([real, f0, f1,
                                   real_power, real_subtraction, real_addition, real_multiplication])
    
    explorationCompression(baseGrammar, tasks,
                           outputPrefix = "experimentOutputs/scientificLaws",
                           evaluationTimeout = 0.1,
                           testingTasks = [],
                           **commandlineArguments(
                               iterations = 10,
                                                  CPUs = numberOfCPUs(),
                                                  structurePenalty = 1.,
                                                  helmholtzRatio = 0.5,
                                                  a = 1,#arity
                                                  maximumFrontier = 1000,
                                                  topK = 2,
                                                  featureExtractor = None,
                                                  pseudoCounts = 10.0))
