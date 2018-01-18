
from utilities import eprint
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
                 taskSolutions = {},
                 parameters = None):
        self.parameters = parameters
        self.learningCurve = learningCurve
        self.grammars = grammars
        self.taskSolutions = taskSolutions
        

def explorationCompression(primitives, tasks,
                           _ = None,
                           iterations = None,
                           frontierSize = None,
                           useRecognitionModel = True,
                           topK = 1,
                           maximumFrontier = None,
                           pseudoCounts = 1.0, aic = 1.0, structurePenalty = 0.001, arity = 0,
                           CPUs = 1,
                           outputPrefix = None):
    if frontierSize == None:
        eprint("Please specify a frontier size: explorationCompression(..., frontierSize = ...)")
        assert False
    if iterations == None:
        eprint("Please specify a iteration count: explorationCompression(..., iterations = ...)")
        assert False
    if useRecognitionModel and not all( len(t.features) == len(tasks[0].features) for t in tasks ):
        eprint("Warning: Recognition model needs features to all have the same dimensionality. Ignoring recognition model.")
        useRecognitionModel = False


    # We save the parameters that were passed into EC
    # This is for the purpose of exporting the results of the experiment
    parameters = {k: v for k,v in locals().iteritems()
                  if not k in ['tasks','primitives','_','CPUs','outputPrefix'] }
    
    grammar = Grammar.uniform(primitives)

    grammarHistory = [grammar]
    learningCurve = []

    for j in range(iterations):
        frontiers = callCompiled(enumerateFrontiers,
                                 grammar, frontierSize, tasks,
                                 maximumFrontier = maximumFrontier,
                                 CPUs = CPUs)
        
        eprint("Enumeration results:")
        eprint(Frontier.describe(frontiers))

        if useRecognitionModel:
            # Train and then use a recognition model
            recognizer = RecognitionModel(len(tasks[0].features), grammar)
            recognizer.train(frontiers, topK = topK)
            bottomupFrontiers = recognizer.enumerateFrontiers(frontierSize, tasks)
            eprint("Bottom-up enumeration results:")
            eprint(Frontier.describe(bottomupFrontiers))

            # Rescore the frontiers according to the generative model and then combine w/ original frontiers
            generativeModel = FragmentGrammar.fromGrammar(grammar)
            bottomupFrontiers = [ generativeModel.rescoreFrontier(f) for f in bottomupFrontiers ]

            frontiers = [ f.combine(b) for f,b in zip(frontiers, bottomupFrontiers) ]
            

        # number of hit tasks
        learningCurve.append(sum(not f.empty for f in frontiers))


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
        eprint("Grammar after iteration %d:"%(j+1))
        eprint(grammar)

    returnValue = ECResult(learningCurve = learningCurve,
                           grammars = grammarHistory,
                           parameters = parameters,
                           taskSolutions = {f.task: f.bestPosterior
                                     for f in frontiers if not f.empty })

    if outputPrefix is not None:
        path = outputPrefix + "_" + \
               "_".join(k + "=" + str(parameters[k]) for k in sorted(parameters.keys()) ) + ".pickle"
        with open(path, 'wb') as handle:
            pickle.dump(returnValue, handle)
        eprint("Exported experiment result to",path)

    return returnValue

        
def commandlineArguments(_ = None,
                         iterations = None,
                         frontierSize = None,
                         topK = 1,
                         CPUs = 1,
                         useRecognitionModel = True,
                         maximumFrontier = None,
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
    parser.add_argument("-m","--maximumFrontier",
                        help = "Even though we enumerate --frontierSize programs, we might want to only keep around the very best for performance reasons. This is a cut off on the maximum size of the frontier that is kept around. default %s"%maximumFrontier,
                        type = int)
    parser.add_argument("-r", "--recognition",
                        dest = 'useRecognitionModel',
                        action = 'store_true',
                        help = "Enable bottom-up neural recognition model. Default: %s"%useRecognitionModel)
    parser.add_argument("-g", "--no-recognition",
                        dest = 'useRecognitionModel',
                        action = 'store_false',
                        help = "Disable bottom-up neural recognition model. Default: %s"%(not useRecognitionModel))
    parser.set_defaults(useRecognitionModel = useRecognitionModel)
    return vars(parser.parse_args())
    
    
