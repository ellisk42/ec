
from recognition import *
from frontier import *
from program import *
from type import *
from task import *
from enumeration import *
from grammar import *
from fragmentGrammar import *

class ECResult():
    def __init__(self, _ = None,
                 learningCurve = [],
                 grammars = [],
                 taskSolutions = {}):
        self.learningCurve = learningCurve
        self.grammars = grammars
        self.taskSolutions = taskSolutions
        

def explorationCompression(primitives, tasks,
                           _ = None,
                           iterations = None,
                           frontierSize = None,
                           topK = 1,
                           pseudoCounts = 1.0, aic = 1.0, structurePenalty = 0.001, arity = 0,
                           CPUs = 1):
    if frontierSize == None:
        print "Please specify a frontier size: explorationCompression(..., frontierSize = ...)"
        assert False
    if iterations == None:
        print "Please specify a iteration count: explorationCompression(..., iterations = ...)"
        assert False

    grammar = Grammar.uniform(primitives)

    grammarHistory = [grammar]
    learningCurve = []

    for j in range(iterations):
        frontiers = callCompiled(enumerateFrontiers,
                                 grammar, frontierSize, tasks,
                                 CPUs = CPUs)
        
        print "Enumeration results:"
        print Frontier.describe(frontiers)

        # number of hit tasks
        learningCurve.append(sum(not f.empty for f in frontiers))

        if False:
            recognizer = RecognitionModel(len(tasks[0].features), grammar)
            recognizer.train(frontiers)
            bottomFrontiers = recognizer.enumerateFrontiers(frontierSize, tasks)
            print "Bottom-up enumeration results:"
            print Frontier.describe(frontiers)

        grammar = callCompiled(induceFragmentGrammarFromFrontiers,
                               grammar,
                               frontiers,
                               topK = topK,
                               pseudoCounts = pseudoCounts,
                               aic = aic,
                               structurePenalty = structurePenalty,
                               a = arity,
                               CPUs = CPUs).\
                               toGrammar()
        grammarHistory.append(grammar)
        print "Final grammar:"
        print grammar

    return ECResult(learningCurve = learningCurve,
                    grammars = grammarHistory,
                    taskSolutions = {f.task: f.bestPosterior
                                     for f in frontiers if not f.empty })

        
def commandlineArguments(_ = None,
                         iterations = None,
                         frontierSize = None,
                         topK = 1,
                         CPUs = 1,
                         pseudoCounts = 1.0, aic = 1.0, structurePenalty = 0.001, a = 0):
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument('-i',"--iterations",
                        help = 'default %d'%iterations,
                        default = iterations,
                        type = int)
    parser.add_argument("-f","--frontierSize",
                        default = frontierSize,
                        help = 'default %d'%frontierSize,
                        type = int)
    parser.add_argument('-k',"--topK",
                        default = topK,
                        help = 'default %d'%topK,
                        type = int)
    parser.add_argument("-p","--pseudoCounts",
                        default = pseudoCounts,
                        help = 'default %f'%pseudoCounts,
                        type = float)
    parser.add_argument("-b","--aic",
                        default = aic,
                        help = 'default %f'%aic,
                        type = float)
    parser.add_argument("-l", "--structurePenalty",
                        default = structurePenalty,
                        help = 'default %f'%structurePenalty,
                        type = float)
    parser.add_argument("-a", "--arity",
                        default = a,
                        help = 'default %d'%a,
                        type = int)
    parser.add_argument("-c", "--CPUs",
                        default = CPUs,
                        help = 'default %d'%CPUs,
                        type = int)
    return vars(parser.parse_args())
    
    
