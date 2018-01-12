
from recognition import *
from frontier import *
from program import *
from type import *
from task import *
from enumeration import *
from grammar import *
from fragmentGrammar import *


def explorationCompression(primitives, tasks,
                           _ = None,
                           iterations = None,
                           frontierSize = None,
                           topK = 1,
                           pseudoCounts = 1.0, aic = 1.0, structurePenalty = 0.001, a = 0):
    if frontierSize == None:
        print "Please specify a frontier size: explorationCompression(..., frontierSize = ...)"
        assert False
    if iterations == None:
        print "Please specify a iteration count: explorationCompression(..., iterations = ...)"
        assert False

    grammar = Grammar.uniform(primitives)

    for j in range(iterations):
        frontiers = callCompiled(enumerateFrontiers,
                                 grammar, frontierSize, tasks)
        frontiers = [ frontier.keepTopK(1) for frontier in frontiers ]

        recognizer = RecognitionModel(len(tasks[0].features), grammar)
        recognizer.train(frontiers)
        bottomFrontiers = [ f.keepTopK(1) for f in recognizer.enumerateFrontiers(frontierSize, tasks)
                            if not f.empty() ]
        print "Bottom-up enumeration hits %d tasks"%(len(bottomFrontiers))
        print "Of these it has an average log likelihood of", sum(f.bestPosterior().logPrior for f in bottomFrontiers )/len(bottomFrontiers)

        numberOfHitTasks = 0
        for frontier in frontiers:
            if frontier.empty():
                #print "MISS",frontier.task.name
                pass
            else:
                print "HIT",frontier.task.name,"with",frontier.bestPosterior().program,frontier.bestPosterior().logPrior
                numberOfHitTasks += 1
        print "Hit %d/%d tasks"%(numberOfHitTasks,len(tasks))
        print "Of these it has an average log likelihood of",sum(f.bestPosterior().logPrior for f in frontiers if not f.empty() )/numberOfHitTasks

        grammar = callCompiled(FragmentGrammar.induceFromFrontiers,
                               grammar, frontiers,
                           pseudoCounts = pseudoCounts, aic = aic, structurePenalty = structurePenalty, a = 0).\
                           toGrammar()
        print grammar

        
