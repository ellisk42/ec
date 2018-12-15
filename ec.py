
from utilities import eprint
from likelihoodModel import *
from recognition import *
from frontier import *
from program import *
from type import *
from task import *
from enumeration import *
from grammar import *
from fragmentGrammar import *
from taskBatcher import *
import baselines
import dill


import os
import datetime

import pickle as pickle

import torch


class ECResult():
    def __init__(self, _=None,
                 testingSearchTime=None,
                 learningCurve=None,
                 grammars=None,
                 taskSolutions=None,
                 averageDescriptionLength=None,
                 parameters=None,
                 recognitionModel=None,
                 searchTimes=None,
                 recognitionTaskTimes=None,
                 recognitionTaskMetrics=None,
                 baselines=None,
                 numTestingTasks=None,
                 sumMaxll=None,
                 testingSumMaxll=None,
                 hitsAtEachWake=None,
                 timesAtEachWake=None,
                 allFrontiers=None):
        self.hitsAtEachWake = hitsAtEachWake or []
        self.timesAtEachWake = timesAtEachWake or []
        self.testingSearchTime = testingSearchTime or []
        self.searchTimes = searchTimes or []
        self.recognitionTaskTimes = recognitionTaskTimes or {}
        self.recognitionTaskMetrics = recognitionTaskMetrics or {} 
        self.recognitionModel = recognitionModel
        self.averageDescriptionLength = averageDescriptionLength or []
        self.parameters = parameters
        self.learningCurve = learningCurve or []
        self.grammars = grammars or []
        self.taskSolutions = taskSolutions or {}
        # baselines is a dictionary of name -> ECResult
        self.baselines = baselines or {}
        self.numTestingTasks = numTestingTasks
        self.sumMaxll = sumMaxll or [] #TODO name change 
        self.testingSumMaxll = testingSumMaxll or [] #TODO name change
        self.allFrontiers = allFrontiers or {}

    def __repr__(self):
        attrs = ["{}={}".format(k, v) for k, v in self.__dict__.items()]
        return "ECResult({})".format(", ".join(attrs))

    # Linux does not like files that have more than 256 characters
    # So when exporting the results we abbreviate the parameters
    abbreviations = {"frontierSize": "fs",
                     "reuseRecognition": "RR",
                     "recognitionTimeout": "RT",
                     "iterations": "it",
                     "maximumFrontier": "MF",
                     "onlyBaselines": "baseline",
                     "pseudoCounts": "pc",
                     "structurePenalty": "L",
                     "helmholtzRatio": "HR",
                     "biasOptimal": "BO",
                     "contextual": "CO",
                     "topK": "K",
                     "enumerationTimeout": "ET",
                     "useRecognitionModel": "rec",
                     "useNewRecognitionModel": "newRec",
                     "likelihoodModel": "likemod",
                     "helmholtzBatch": "HB",
                     "use_ll_cutoff": "llcut",
                     "topk_use_only_likelihood": "topkNotMAP",
                     "activation": "act",
                     "storeTaskMetrics": 'storeTask',
                     "rewriteTaskMetrics": "RW",
                     'taskBatchSize': 'batch'}

    @staticmethod
    def abbreviate(parameter): return ECResult.abbreviations.get(parameter, parameter)

    @staticmethod
    def parameterOfAbbreviation(abbreviation):
        return ECResult.abbreviationToParameter.get(abbreviation, abbreviation)

    @staticmethod
    def clearRecognitionModel(path):
        SUFFIX = '.pickle'
        assert path.endswith(SUFFIX)
        
        with open(path,'rb') as handle:
            result = dill.load(handle)
        
        result.recognitionModel = None
        
        clearedPath = path[:-len(SUFFIX)] + "_graph=True" + SUFFIX
        with open(clearedPath,'wb') as handle:
            result = dill.dump(result, handle)
        eprint(" [+] Cleared recognition model from:")
        eprint("     %s"%path)
        eprint("     and exported to:")
        eprint("     %s"%clearedPath)
        eprint("     Use this one for graphing.")


ECResult.abbreviationToParameter = {
    v: k for k, v in ECResult.abbreviations.items()}


def explorationCompression(*arguments, **keywords):
    for r in ecIterator(*arguments, **keywords):
        pass
    return r


def ecIterator(grammar, tasks,
               _=None,
               bootstrap=None,
               solver="ocaml",
               compressor="rust",
               likelihoodModel="all-or-nothing",
               biasOptimal=False,
               contextual=False,
               testingTasks=[],
               benchmark=None,
               iterations=None,
               resume=None,
               frontierSize=None,
               enumerationTimeout=None,
               testingTimeout=None,
               testEvery=1,
               reuseRecognition=False,
               expandFrontier=None,
               resumeFrontierSize=None,
               useRecognitionModel=True,
               useNewRecognitionModel=False,
               recognitionTimeout=None,
               helmholtzRatio=0.,
               helmholtzBatch=5000,
               featureExtractor=None,
               activation='relu',
               topK=1,
               topk_use_only_likelihood=False,
               use_map_search_times=True,
               maximumFrontier=None,
               pseudoCounts=1.0, aic=1.0,
               structurePenalty=0.001, arity=0,
               evaluationTimeout=1.0,  # seconds
               taskBatchSize=None,
               taskReranker='default',
               CPUs=1,
               cuda=False,
               message="",
               onlyBaselines=False,
               outputPrefix=None,
               storeTaskMetrics=False,
               rewriteTaskMetrics=True):
    if frontierSize is None and enumerationTimeout is None:
        eprint(
            "Please specify a frontier size and/or an enumeration timeout:",
            "explorationCompression(..., enumerationTimeout = ..., frontierSize = ...)")
        assert False
    if iterations is None:
        eprint(
            "Please specify a iteration count: explorationCompression(..., iterations = ...)")
        assert False
    if useRecognitionModel and featureExtractor is None:
        eprint("Warning: Recognition model needs feature extractor.",
               "Ignoring recognition model.")
        useRecognitionModel = False
    if useNewRecognitionModel and featureExtractor is None:
        eprint("Warning: Recognition model needs feature extractor.",
               "Ignoring recognition model.")
        useNewRecognitionModel = False
    if benchmark is not None and resume is None:
        eprint("You cannot benchmark unless you are loading a checkpoint, aborting.")
        assert False
    if biasOptimal and not useRecognitionModel:
        eprint("Bias optimality only applies to recognition models, aborting.")
        assert False
    if contextual and not useRecognitionModel:
        eprint("Contextual only applies to recognition models, aborting")
        assert False
    if reuseRecognition and not useRecognitionModel:
        eprint("Reuse of recognition model weights at successive iteration only applies to recognition models, aborting")
        assert False

    if testingTimeout > 0 and len(testingTasks) == 0:
        eprint("You specified a testingTimeout, but did not provide any held out testing tasks, aborting.")
        assert False

    if expandFrontier is not None:
        if taskReranker is not 'default' or taskBatchSize is not None:
            eprint("You specified batched tasks, which is not compatible with frontier expansion, aborting.")
            assert False


    # We save the parameters that were passed into EC
    # This is for the purpose of exporting the results of the experiment
    parameters = {
        k: v for k,
        v in locals().items() if k not in {
            "tasks",
            "useNewRecognitionModel",
            "likelihoodModel",
            "use_map_search_times",
            "activation",
            "helmholtzBatch",
            "grammar",
            "cuda",
            "_",
            "solver",
            "testingTimeout",
            "testEvery",
            "message",
            "CPUs",
            "outputPrefix",
            "resume",
            "resumeFrontierSize",
            "bootstrap",
            "featureExtractor",
            "benchmark",
            "evaluationTimeout",
            "testingTasks",
            "compressor"} and v is not None}
    if not useRecognitionModel:
        for k in {"helmholtzRatio", "recognitionTimeout", "biasOptimal", "contextual", "reuseRecognition"}:
            if k in parameters: del parameters[k]

    # Uses `parameters` to construct the checkpoint path
    def checkpointPath(iteration, extra=""):
        parameters["iterations"] = iteration
        kvs = [
            "{}={}".format(
                ECResult.abbreviate(k),
                parameters[k]) for k in sorted(
                parameters.keys())]
        if useRecognitionModel or useNewRecognitionModel:
            kvs += ["feat=%s" % (featureExtractor.__name__)]
        if bootstrap:
            kvs += ["bstrap=True"]
        return "{}_{}{}.pickle".format(outputPrefix, "_".join(kvs), extra)

    if onlyBaselines and not benchmark:
        result = ECResult()
        result.baselines = baselines.all(
            grammar,
            tasks,
            CPUs=CPUs,
            cuda=cuda,
            featureExtractor=featureExtractor,
            compressor=compressor,
            **parameters)
        if outputPrefix is not None:
            path = checkpointPath(0, extra="_e")
            with open(path, "wb") as f:
                pickle.dump(result, f)
            eprint("Exported checkpoint to", path)
        yield result
        return

    if message:
        message = " (" + message + ")"
    eprint("Running EC%s on %s @ %s with %d CPUs and parameters:" %
           (message, os.uname()[1], datetime.datetime.now(), CPUs))
    for k, v in parameters.items():
        eprint("\t", k, " = ", v)
    eprint("\t", "evaluationTimeout", " = ", evaluationTimeout)
    eprint("\t", "cuda", " = ", cuda)
    eprint()


    # Restore checkpoint
    if resume is not None:
        try:
            resume = int(resume)
            path = checkpointPath(resume, extra="_baselines" if onlyBaselines else "")
        except ValueError:
            path = resume
        with open(path, "rb") as handle:
            result = dill.load(handle)
        resume = len(result.grammars) - 1
        eprint("Loaded checkpoint from", path)
        grammar = result.grammars[-1] if result.grammars else grammar
        recognizer = result.recognitionModel
        if resumeFrontierSize:
            frontierSize = resumeFrontierSize
            eprint("Set frontier size to", frontierSize)
        if bootstrap is not None:  # Make sure that we register bootstrapped primitives
            for p in grammar.primitives:
                RegisterPrimitives.register(p)
    else:  # Start from scratch
        if bootstrap is not None:
            with open(bootstrap, "rb") as handle:
                strapping = pickle.load(handle).grammars[-1]
            eprint("Bootstrapping from", bootstrap)
            eprint("Bootstrap primitives:")
            for p in strapping.primitives:
                eprint(p)
                RegisterPrimitives.register(p)
            eprint()
            grammar = Grammar.uniform(list({p for p in grammar.primitives + strapping.primitives
                                            if not str(p).startswith("fix")}),
                                      continuationType=grammar.continuationType)
            if compressor == "rust":
                eprint(
                    "Rust compressor is currently not compatible with bootstrapping.",
                    "Falling back on pypy compressor.")
                compressor = "pypy"

        #for graphing of testing tasks
        numTestingTasks = len(testingTasks) if len(testingTasks) != 0 else None

        result = ECResult(parameters=parameters,            
                          grammars=[grammar],
                          taskSolutions={
                              t: Frontier([],
                                          task=t) for t in tasks},
                          recognitionModel=None, numTestingTasks=numTestingTasks,
                          allFrontiers={
                              t: Frontier([],
                                          task=t) for t in tasks})


    if benchmark is not None:
        assert resume is not None, "Benchmarking requires resuming from checkpoint that you are benchmarking."
        if benchmark > 0:
            assert testingTasks != [], "Benchmarking requires held out test tasks"
            benchmarkTasks = testingTasks
        else:
            benchmarkTasks = tasks
            benchmark = -benchmark
        if len(result.baselines) == 0:
            results = {"our algorithm": result}
        else:
            results = result.baselines
        for name, result in results.items():
            eprint("Starting benchmark:", name)
            benchmarkSynthesisTimes(
                result,
                benchmarkTasks,
                timeout=benchmark,
                CPUs=CPUs,
                evaluationTimeout=evaluationTimeout)
            eprint("Completed benchmark.")
            eprint()
        yield None
        return


    likelihoodModel = {
        "all-or-nothing": lambda: AllOrNothingLikelihoodModel(
            timeout=evaluationTimeout),
        "feature-discriminator": lambda: FeatureDiscriminatorLikelihoodModel(
            tasks,
            featureExtractor(tasks)),
        "euclidean": lambda: EuclideanLikelihoodModel(
            featureExtractor(tasks)),
        "probabilistic": lambda: ProbabilisticLikelihoodModel(
            timeout=evaluationTimeout)}[likelihoodModel]()

    # Set up the task batcher.
    if taskReranker == 'default':
        taskBatcher = DefaultTaskBatcher()
    elif taskReranker == 'random':
        taskBatcher = RandomTaskBatcher()
    elif taskReranker == 'randomShuffle':
        taskBatcher = RandomShuffleTaskBatcher()
    elif taskReranker == 'unsolved':
        taskBatcher = UnsolvedTaskBatcher()
    elif taskReranker == 'unsolvedEntropy':
        taskBatcher = UnsolvedEntropyTaskBatcher()
    elif taskReranker == 'unsolvedRandomEntropy':
        taskBatcher = UnsolvedRandomEntropyTaskBatcher()
    elif taskReranker == 'randomkNN':
        taskBatcher = RandomkNNTaskBatcher()
    elif taskReranker == 'randomLowEntropykNN':
        taskBatcher = RandomLowEntropykNNTaskBatcher()
    else:
        eprint("Invalid task reranker: " + taskReranker + ", aborting.")
        assert False
 
    for j in range(resume or 0, iterations):
        if storeTaskMetrics and rewriteTaskMetrics:
            eprint("Resetting task metrics for next iteration.")
            result.recognitionTaskMetrics = {}

        # Evaluate on held out tasks if we have them
        if testingTimeout > 0 and ((j % testEvery == 0) or (j == iterations - 1)):
            eprint("Evaluating on held out testing tasks for iteration: %d" % (j))
            if useRecognitionModel and result.recognitionModel is not None: 
                eprint("Evaluating using trained recognition model.")
                testingFrontiers, times, testingTimes = result.recognitionModel.enumerateFrontiers(testingTasks, likelihoodModel,
                                                                      CPUs=CPUs,
                                                                      solver=solver,
                                                                      maximumFrontier=maximumFrontier,
                                                                      enumerationTimeout=testingTimeout,
                                                                      evaluationTimeout=evaluationTimeout,
                                                                      testing=True)

                if storeTaskMetrics:
                    updateTaskSummaryMetrics(result.recognitionTaskMetrics, testingTimes, 'heldoutTestingTimes')
                    updateTaskSummaryMetrics(result.recognitionTaskMetrics, recognizer.taskGrammarLogProductions(testingTasks), 'heldoutTaskLogProductions')
                    updateTaskSummaryMetrics(result.recognitionTaskMetrics, recognizer.taskGrammarEntropies(testingTasks), 'heldoutTaskGrammarEntropies')


            else:
                eprint("Evaluating using multicore enumeration without a recognition model.")
                testingFrontiers, times, allTimes = multicoreEnumeration(grammar, testingTasks, likelihoodModel,
                                                               solver=solver,
                                                               maximumFrontier=maximumFrontier,
                                                               enumerationTimeout=testingTimeout,
                                                               CPUs=CPUs,
                                                               evaluationTimeout=evaluationTimeout,
                                                               testing=True)
            print("\n".join(f.summarize() for f in testingFrontiers))
            eprint("Hits %d/%d testing tasks" % (len(times), len(testingTasks)))

            summaryStatistics("Testing tasks", times)
            result.testingSearchTime.append(times)
            result.testingSumMaxll.append(sum(math.exp(f.bestll) for f in testingFrontiers if not f.empty) )

            
        # If we have to also enumerate Helmholtz frontiers,
        # do this extra sneaky in the background
        if useRecognitionModel and biasOptimal and helmholtzRatio > 0 and \
           all( str(p) != "REAL" for p in grammar.primitives ): # real numbers don't support this
            helmholtzFrontiers = backgroundHelmholtzEnumeration(tasks, grammar, enumerationTimeout,
                                                                evaluationTimeout=evaluationTimeout,
                                                                special=featureExtractor.special)
        else:
            helmholtzFrontiers = lambda: []

        # Get waking task batch.
        wakingTaskBatch = taskBatcher.getTaskBatch(result, tasks, taskBatchSize, j)
        eprint("Using a waking task batch of size: " + str(len(wakingTaskBatch)))

        # WAKING UP
        topDownFrontiers, times, allTimes = multicoreEnumeration(grammar, wakingTaskBatch, likelihoodModel,
                                                solver=solver,
                                                maximumFrontier=maximumFrontier,
                                                enumerationTimeout=enumerationTimeout,
                                                CPUs=CPUs,
                                                evaluationTimeout=evaluationTimeout)

        eprint("Generative model enumeration results:")
        eprint(Frontier.describe(topDownFrontiers))
        summaryStatistics("Generative model", times)

        tasksHitTopDown = {f.task for f in topDownFrontiers if not f.empty}
        result.hitsAtEachWake.append(len(tasksHitTopDown))
        #result.timesAtEachWake.append(times)

        # Combine topDownFrontiers from this task batch with all frontiers.
        for f in topDownFrontiers:
            result.allFrontiers[f.task] = result.allFrontiers[f.task].combine(f).topK(maximumFrontier)

        eprint("Frontiers discovered top down: " + str(len(tasksHitTopDown)))
        eprint("Total frontiers: " + str(len([f for f in result.allFrontiers.values() if not f.empty])))

        # Train + use recognition model
        if useRecognitionModel:
            if len([f for f in result.allFrontiers.values() if not f.empty]) == 0:
                eprint("No frontiers to train recognition model, cannot do recognition model enumeration.")
                tasksHitBottomUp = set()
            else:
                # Should we initialize the weights to be what they were before?
                previousRecognitionModel = None
                if reuseRecognition and result.recognitionModel is not None:
                    previousRecognitionModel = result.recognitionModel
                
                featureExtractorObject = featureExtractor(tasks, testingTasks=testingTasks, cuda=cuda)
                recognizer = RecognitionModel(featureExtractorObject,
                                              grammar,
                                              activation=activation,
                                              cuda=cuda,
                                              contextual=contextual,
                                              previousRecognitionModel=previousRecognitionModel)
                
                thisRatio = helmholtzRatio
                if j == 0 and not biasOptimal: thisRatio = 0
                recognizer.train(result.allFrontiers.values(),
                                 biasOptimal=biasOptimal,
                                 helmholtzFrontiers=helmholtzFrontiers(), 
                                 CPUs=CPUs,
                                 evaluationTimeout=evaluationTimeout,
                                 timeout=recognitionTimeout,
                                 helmholtzRatio=thisRatio)
                
                result.recognitionModel = recognizer

                bottomupFrontiers, times, allRecognitionTimes = recognizer.enumerateFrontiers(wakingTaskBatch, likelihoodModel,
                                                                         CPUs=CPUs,
                                                                         solver=solver,
                                                                         frontierSize=frontierSize,
                                                                         maximumFrontier=maximumFrontier,
                                                                         enumerationTimeout=enumerationTimeout,
                                                                         evaluationTimeout=evaluationTimeout)
                # Store the recognition metrics.
                if storeTaskMetrics:
                    updateTaskSummaryMetrics(result.recognitionTaskMetrics, allRecognitionTimes, 'recognitionBestTimes')
                    updateTaskSummaryMetrics(result.recognitionTaskMetrics, recognizer.taskGrammarLogProductions(wakingTaskBatch), 'taskLogProductions')
                    updateTaskSummaryMetrics(result.recognitionTaskMetrics, recognizer.taskGrammarEntropies(wakingTaskBatch), 'taskGrammarEntropies')

                tasksHitBottomUp = {f.task for f in bottomupFrontiers if not f.empty}
                #result.timesAtEachWake.append(times)
                result.hitsAtEachWake.append(len(tasksHitBottomUp))
                

        elif useNewRecognitionModel:  # Train a recognition model
            if len([f for f in result.allFrontiers.values() if not f.empty]) == 0:
                eprint("No frontiers to train recognition model, cannot do recognition model enumeration.")
                tasksHitBottomUp = set()
            else:
                result.recognitionModel.updateGrammar(grammar)
                result.recognitionModel.train(
                    result.allFrontiers.values(),
                    topK=topK,
                    helmholtzRatio=helmholtzRatio)
                eprint("done training recognition model")
                bottomupFrontiers, times, allRecognitionTimes = result.recognitionModel.enumerateFrontiers(
                    wakingTaskBatch,
                    likelihoodModel,
                    CPUs=CPUs,
                    solver=solver,
                    maximumFrontier=maximumFrontier,
                    frontierSize=frontierSize,
                    enumerationTimeout=enumerationTimeout,
                    evaluationTimeout=evaluationTimeout)
                tasksHitBottomUp = {f.task for f in bottomupFrontiers if not f.empty}


        # Repeatedly expand the frontier until we hit something that we have not solved yet
        solvedTasks = tasksHitTopDown | (tasksHitBottomUp if useRecognitionModel else set())
        numberOfSolvedTasks = len(solvedTasks)
        if j > 0 and expandFrontier and numberOfSolvedTasks <= result.learningCurve[-1] and \
           result.learningCurve[-1] < len(tasks):
   
            # Focus on things we did not solve this iteration AND also did not solve last iteration
            unsolved = [t for t in tasks if (t not in solvedTasks) and result.taskSolutions[t].empty ]
            eprint("We are currently stuck: there are %d remaining unsolved tasks, and we only solved %d ~ %d in the last two iterations"%(len(unsolved),
                                                                                                                                         numberOfSolvedTasks,
                                                                                                                                         result.learningCurve[-1]))
            eprint("Going to repeatedly expand the search timeout until we solve something new...")
            timeout = enumerationTimeout
            while True:
                eprint("Expanding enumeration timeout from %i to %i because of no progress. Focusing exclusively on %d unsolved tasks." % (timeout, timeout * expandFrontier, len(unsolved)))
                timeout = timeout * expandFrontier
                unsolvedFrontiers, unsolvedTimes, allUnsolvedTimes = \
                    multicoreEnumeration(grammar, unsolved, likelihoodModel,
                                         solver=solver,
                                         maximumFrontier=maximumFrontier,
                                         enumerationTimeout=timeout,
                                         CPUs=CPUs,
                                         evaluationTimeout=evaluationTimeout)
                if useRecognitionModel:
                    bottomUnsolved, unsolvedTimes, allUnsolvedRecognitionTimes = recognizer.enumerateFrontiers(unsolved, likelihoodModel,
                                                                                  CPUs=CPUs,
                                                                                  solver=solver,
                                                                                  frontierSize=frontierSize,
                                                                                  maximumFrontier=maximumFrontier,
                                                                                  enumerationTimeout=timeout,
                                                                                  evaluationTimeout=evaluationTimeout)
                   
                    # Merge top-down w/ bottom-up
                    unsolvedFrontiers = [f.combine(grammar.rescoreFrontier(b))
                                         for f, b in zip(unsolvedFrontiers, bottomUnsolved) ]
                    
                if any(not f.empty for f in unsolvedFrontiers):
                    times += unsolvedTimes
                    unsolvedFrontiers = {f.task: f for f in unsolvedFrontiers}
                    frontiers = [f if (not f.empty) or (f.task not in unsolvedFrontiers) \
                                 else unsolvedFrontiers[f.task]
                                 for f in frontiers]
                    print("Completed frontier expansion; solved: %s"%
                          {t.name for t,f in unsolvedFrontiers.items() if not f.empty })
                    break
                
        if useRecognitionModel or useNewRecognitionModel:
            if len([f for f in result.allFrontiers.values() if not f.empty]) > 0:
                eprint("Recognition model enumeration results:")
                eprint(Frontier.describe(bottomupFrontiers))
                summaryStatistics("Recognition model", times)

                result.averageDescriptionLength.append(mean(-f.marginalLikelihood()
                                                            for f in bottomupFrontiers
                                                            if not f.empty))

                result.sumMaxll.append( sum(math.exp(f.bestll) for f in bottomupFrontiers if not f.empty)) #TODO

                showHitMatrix(tasksHitTopDown, tasksHitBottomUp, wakingTaskBatch)
                # Rescore the frontiers according to the generative model
                # and then combine w/ original frontiers
                for b in bottomupFrontiers:
                    result.allFrontiers[b.task] = result.allFrontiers[b.task].\
                                                  combine(grammar.rescoreFrontier(b)).\
                                                  topK(maximumFrontier)

                eprint("Frontiers discovered bottom up: " + str(len(tasksHitBottomUp)))
                eprint("Total frontiers: " + str(len([f for f in result.allFrontiers.values() if not f.empty])))


        else:
            result.averageDescriptionLength.append(mean(-f.marginalLikelihood()
                                                        for f in result.allFrontiers.values()
                                                        if not f.empty))

            result.sumMaxll.append(sum(math.exp(f.bestll) for f in result.allFrontiers.values() if not f.empty)) #TODO - i think this is right

        if not useNewRecognitionModel:  # This line is changed, beware
            result.searchTimes.append(times)
            if len(times) > 0:
                eprint("Average search time: ", int(mean(times) + 0.5),
                       "sec.\tmedian:", int(median(times) + 0.5),
                       "\tmax:", int(max(times) + 0.5),
                       "\tstandard deviation", int(standardDeviation(times) + 0.5))


        eprint("Showing the top 5 programs in each frontier:")
        for f in result.allFrontiers.values():
            if f.empty:
                continue
            eprint(f.task)
            for e in f.normalize().topK(5):
                eprint("%.02f\t%s" % (e.logPosterior, e.program))
            eprint()
            
        # Record the new topK solutions
        result.taskSolutions = {f.task: f.topK(topK)
                                for f in result.allFrontiers.values()}
        result.learningCurve += [
            sum(f is not None and not f.empty for f in result.taskSolutions.values())]
                
        
        # Sleep-G
        # First check if we have supervision at the program level for any task that was not solved
        needToSupervise = {f.task for f in result.allFrontiers.values()
                           if f.task.supervision is not None and f.empty}
        compressionFrontiers = [f.replaceWithSupervised(grammar) if f.task in needToSupervise else f
                                for f in result.allFrontiers.values() ]

        if len([f for f in compressionFrontiers if not f.empty]) == 0:
            eprint("No compression frontiers; not inducing a grammar this iteration.")
        else:
            grammar, compressionFrontiers = induceGrammar(grammar, compressionFrontiers,
                                                          topK=topK,
                                                          pseudoCounts=pseudoCounts, a=arity,
                                                          aic=aic, structurePenalty=structurePenalty,
                                                          topk_use_only_likelihood=topk_use_only_likelihood,
                                                          backend=compressor, CPUs=CPUs, iteration=j)
            # Store compression frontiers in the result.
            for c in compressionFrontiers:
                result.allFrontiers[c.task] = c.topK(0) if c in needToSupervise else c


        result.grammars.append(grammar)
        eprint("Grammar after iteration %d:" % (j + 1))
        eprint(grammar)

        
        if outputPrefix is not None:
            path = checkpointPath(j + 1)
            with open(path, "wb") as handle:
                try:
                    dill.dump(result, handle)
                except TypeError as e:
                    eprint(result)
                    assert(False)
            eprint("Exported checkpoint to", path)
            if useRecognitionModel:
                ECResult.clearRecognitionModel(path)

            graphPrimitives(result, "%s_primitives_%d_"%(outputPrefix,j))
            

        yield result


def showHitMatrix(top, bottom, tasks):
    tasks = set(tasks)

    total = bottom | top
    eprint(len(total), "/", len(tasks), "total hit tasks")
    bottomMiss = tasks - bottom
    topMiss = tasks - top

    eprint("{: <13s}{: ^13s}{: ^13s}".format("", "bottom miss", "bottom hit"))
    eprint("{: <13s}{: ^13d}{: ^13d}".format("top miss",
                                             len(bottomMiss & topMiss),
                                             len(bottom & topMiss)))
    eprint("{: <13s}{: ^13d}{: ^13d}".format("top hit",
                                             len(top & bottomMiss),
                                             len(top & bottom)))


def commandlineArguments(_=None,
                         iterations=None,
                         frontierSize=None,
                         enumerationTimeout=None,
                         testEvery=1,
                         topK=1,
                         reuseRecognition=False,
                         CPUs=1,
                         compressor="ocaml",
                         useRecognitionModel=True,
                         useNewRecognitionModel=False,
                         recognitionTimeout=None,
                         activation='relu',
                         helmholtzRatio=1.,
                         helmholtzBatch=5000,
                         featureExtractor=None,
                         cuda=None,
                         maximumFrontier=None,
                         pseudoCounts=1.0, aic=1.0,
                         structurePenalty=0.001, a=0,
                         taskBatchSize=None, taskReranker="default",
                         onlyBaselines=False,
                         extras=None,
                         storeTaskMetrics=False,
                        rewriteTaskMetrics=True):
    if cuda is None:
        cuda = torch.cuda.is_available()
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--resume",
                        help="Resumes EC algorithm from checkpoint. You can either pass in the path of a checkpoint, or you can pass in the iteration to resume from, in which case it will try to figure out the path.",
                        default=None,
                        type=str)
    parser.add_argument("-i", "--iterations",
                        help="default: %d" % iterations,
                        default=iterations,
                        type=int)
    parser.add_argument("-f", "--frontierSize",
                        default=frontierSize,
                        help="default: %s" % frontierSize,
                        type=int)
    parser.add_argument("-t", "--enumerationTimeout",
                        default=enumerationTimeout,
                        help="In seconds. default: %s" % enumerationTimeout,
                        type=int)
    parser.add_argument("-R", "--recognitionTimeout",
                        default=recognitionTimeout,
                        help="In seconds. Amount of time to train the recognition model on each iteration. Defaults to enumeration timeout.",
                        type=int)
    parser.add_argument(
        "-F",
        "--expandFrontier",
        metavar="FACTOR-OR-AMOUNT",
        default=None,
        help="if an iteration passes where no new tasks have been solved, the frontier is expanded. If the given value is less than 10, it is scaled (e.g. 1.5), otherwise it is grown (e.g. 2000).",
        type=float)
    parser.add_argument(
        "--resumeFrontierSize",
        type=int,
        help="when resuming a checkpoint which expanded the frontier, use this option to set the appropriate frontier size for the next iteration.")
    parser.add_argument(
        "-k",
        "--topK",
        default=topK,
        help="When training generative and discriminative models, we train them to fit the top K programs. Ideally we would train them to fit the entire frontier, but this is often intractable. default: %d" %
        topK,
        type=int)
    parser.add_argument("-p", "--pseudoCounts",
                        default=pseudoCounts,
                        help="default: %f" % pseudoCounts,
                        type=float)
    parser.add_argument("-b", "--aic",
                        default=aic,
                        help="default: %f" % aic,
                        type=float)
    parser.add_argument("-l", "--structurePenalty",
                        default=structurePenalty,
                        help="default: %f" % structurePenalty,
                        type=float)
    parser.add_argument("-a", "--arity",
                        default=a,
                        help="default: %d" % a,
                        type=int)
    parser.add_argument("-c", "--CPUs",
                        default=CPUs,
                        help="default: %d" % CPUs,
                        type=int)
    parser.add_argument("--no-cuda",
                        action="store_false",
                        dest="cuda",
                        help="""cuda will be used if available (which it %s),
                        unless this is set""" % ("IS" if cuda else "ISN'T"))
    parser.add_argument("-m", "--maximumFrontier",
                        help="""Even though we enumerate --frontierSize
                        programs, we might want to only keep around the very
                        best for performance reasons. This is a cut off on the
                        maximum size of the frontier that is kept around.
                        Default: %s""" % maximumFrontier,
                        type=int)
    parser.add_argument("--reuseRecognition",
                        help="""Should we initialize recognition model weights to be what they were at the previous DreamCoder iteration? Default: %s""" % reuseRecognition,
                        default=reuseRecognition,
                        action="store_true")
    parser.add_argument(
        "--benchmark",
        help="""Benchmark synthesis times with a timeout of this many seconds. You must use the --resume option. EC will not run but instead we were just benchmarked the synthesis times of a learned model""",
        type=float,
        default=None)
    parser.add_argument("--recognition",
                        dest="useRecognitionModel",
                        action="store_true",
                        help="""Enable bottom-up neural recognition model.
                        Default: %s""" % useRecognitionModel)
    parser.add_argument("--robustfill",
                        dest="useNewRecognitionModel",
                        action="store_true",
                        help="""Enable bottom-up robustfill recognition model.
                        Default: %s""" % useNewRecognitionModel)
    parser.add_argument("-g", "--no-recognition",
                        dest="useRecognitionModel",
                        action="store_false",
                        help="""Disable bottom-up neural recognition model.
                        Default: %s""" % (not useRecognitionModel))
    parser.add_argument(
        "--testingTimeout",
        type=int,
        dest="testingTimeout",
        default=0,
        help="Number of seconds we should spend evaluating on each held out testing task.")
    parser.add_argument(
        "--testEvery",
        type=int,
        dest="testEvery",
        default=1,
        help="Run heldout testing every X iterations."
        )
    parser.add_argument(
        "--activation",
        choices=[
            "relu",
            "sigmoid",
            "tanh"],
        default=activation,
        help="""Activation function for neural recognition model.
                        Default: %s""" %
        activation)
    parser.add_argument(
        "-r",
        "--Helmholtz",
        dest="helmholtzRatio",
        help="""When training recognition models, what fraction of the training data should be samples from the generative model? Default %f""" %
        helmholtzRatio,
        default=helmholtzRatio,
        type=float)
    parser.add_argument(
        "--helmholtzBatch",
        dest="helmholtzBatch",
        help="""When training recognition models, size of the Helmholtz batch? Default %f""" %
        helmholtzBatch,
        default=helmholtzBatch,
        type=float)
    parser.add_argument(
        "-B",
        "--baselines",
        dest="onlyBaselines",
        action="store_true",
        help="only compute baselines")
    parser.add_argument(
        "--bootstrap",
        help="Start the learner out with a pretrained DSL. This argument should be a path to a checkpoint file.",
        default=None,
        type=str)
    parser.add_argument(
        "--compressor",
        default=compressor,
        choices=["pypy","rust","vs","pypy_vs","ocaml"])
    parser.add_argument("--biasOptimal",
                        help="Enumerate dreams rather than sample them & bias-optimal recognition objective",
                        default=False, action="store_true")
    parser.add_argument("--contextual",
                        help="bigram recognition model (default is unigram model)",
                        default=False, action="store_true")
    parser.add_argument("--clear-recognition",
                        dest="clear-recognition",
                        help="Clears the recognition model from a checkpoint. Necessary for graphing results with recognition models, because pickle is kind of stupid sometimes.",
                        default=None,
                        type=str)
    parser.add_argument("--primitive-graph",
                        dest="primitive-graph",
                        help="Displays a dependency graph of the learned primitives",
                        default=None,
                        type=str)
    parser.add_argument(
        "--taskBatchSize",
        dest="taskBatchSize",
        help="Number of tasks to train on during wake. Defaults to all tasks if None.",
        default=None,
        type=int)
    parser.add_argument(
        "--taskReranker",
        dest="taskReranker",
        help="Reranking function used to order the tasks we train on during waking.",
        choices=[
            "default",
            "random",
            "randomShuffle",
            "unsolved",
            "unsolvedEntropy",
            "unsolvedRandomEntropy",
            "randomkNN",
            "randomLowEntropykNN"],
        default=taskReranker,
        type=str)
    parser.add_argument(
        "--storeTaskMetrics",
        dest="storeTaskMetrics",
        help="Whether to store task metrics directly in the ECResults.",
        action="store_true"
        )
    parser.add_argument(
        "--rewriteTaskMetrics",
        dest="rewriteTaskMetrics",
        help="Whether to rewrite a new task metrics dictionary at each iteration, rather than retaining the old.",
        action="store_true"
        )
    parser.add_argument("--addTaskMetrics",
        dest="addTaskMetrics",
        help="Creates a checkpoint with task metrics and no recognition model for graphing.",
        default=None,
        type=str)
    parser.set_defaults(useRecognitionModel=useRecognitionModel,
                        featureExtractor=featureExtractor,
                        maximumFrontier=maximumFrontier,
                        cuda=cuda)
    if extras is not None:
        extras(parser)
    v = vars(parser.parse_args())
    if v["clear-recognition"] is not None:
        ECResult.clearRecognitionModel(v["clear-recognition"])
        sys.exit(0)
    else:
        del v["clear-recognition"]
        
    if v["primitive-graph"] is not None:
        result = loadPickle(v["primitive-graph"])
        graphPrimitives(result,v["primitive-graph"],view=True)
        sys.exit(0)
    else:
        del v["primitive-graph"]

    if v["addTaskMetrics"] is not None:
        with open(v["addTaskMetrics"],'rb') as handle:
            result = dill.load(handle)
        addTaskMetrics(result, v["addTaskMetrics"])
        sys.exit(0)
    else:
        del v["addTaskMetrics"]

    if v["useRecognitionModel"] and v["recognitionTimeout"] is None:
        v["recognitionTimeout"] = v["enumerationTimeout"]
        
    return v

def addTaskMetrics(result, path):
    """Adds a task metrics to ECResults that were pickled without them."""
    SUFFIX = '.pickle'
    assert path.endswith(SUFFIX)

    tasks = result.taskSolutions.keys()
    eprint("Found %d tasks: " % len(tasks))
    if not hasattr(result, "recognitionTaskMetrics") or result.recognitionTaskMetrics is None:
        result.recognitionTaskMetrics = {}

    # If task has images, store them.
    if hasattr(list(tasks)[0], 'getImage'):
        images = {t: t.getImage(pretty=True) for t in tasks}
        updateTaskSummaryMetrics(result.recognitionTaskMetrics, images, 'taskImages')

    if hasattr(list(tasks)[0], 'highresolution'):
        images = {t: t.highresolution for t in tasks}
        updateTaskSummaryMetrics(result.recognitionTaskMetrics, images, 'taskImages')

    updateTaskSummaryMetrics(result.recognitionTaskMetrics, result.recognitionModel.taskGrammarLogProductions(tasks), 'contextualLogProductions')

    #updateTaskSummaryMetrics(result.recognitionTaskMetrics, result.recognitionModel.taskGrammarLogProductions(tasks), 'task_no_parent_log_productions')
    #updateTaskSummaryMetrics(result.recognitionTaskMetrics, result.recognitionModel.taskGrammarEntropies(tasks), 'taskGrammarEntropies')

    result.recognitionModel = None
        
    clearedPath = path[:-len(SUFFIX)] + "_graph=True" + SUFFIX
    with open(clearedPath,'wb') as handle:
        result = dill.dump(result, handle)
    eprint(" [+] Cleared recognition model from:")
    eprint("     %s"%path)
    eprint("     and exported to:")
    eprint("     %s"%clearedPath)
    eprint("     Use this one for graphing.")

def graphPrimitives(result, prefix, view=False):
    try:
        from graphviz import Digraph
    except:
        eprint("You are missing the graphviz library - cannot graph primitives!")
        return
    

    primitives = { p
                   for g in result.grammars
                   for p in g.primitives
                   if p.isInvented }
    age = {p: min(j for j,g in enumerate(result.grammars) if p in g.primitives)
           for p in primitives }



    ages = set(age.values())
    age2primitives = {a: {p for p,ap in age.items() if a == ap }
                      for a in ages}

    def lb(s,T=20):
        s = s.split()
        l = []
        n = 0
        for w in s:
            if n + len(w) > T:
                l.append("<br />")
                n = 0
            n += len(w)
            l.append(w)
        return " ".join(l)
                
    name = {}
    simplification = {}
    depth = {}
    def getName(p):
        if p in name: return name[p]
        children = {k: getName(k)
                    for _,k in p.body.walk()
                    if k.isInvented}
        simplification_ = p.body
        for k,childName in children.items():
            simplification_ = simplification_.substitute(k, Primitive(childName,None,None))
        name[p] = "f%d"%len(name)
        simplification[p] = name[p] + '=' + lb(str(simplification_))
        depth[p] = 1 + max([depth[k] for k in children] + [0])
        return name[p]

    for p in primitives:
        getName(p)

    depths = {depth[p] for p in primitives}
    depth2primitives = {d: {p for p in primitives if depth[p] == d }
                        for d in depths}

    englishDescriptions = {"#(lambda (lambda (map (lambda (index $0 $2)) (range $0))))":
                           "Prefix",
                           "#(lambda (lambda (fold $0 $1 (lambda (lambda (cons $1 $0))))))":
                           "Append",
                           "#(lambda (cons LPAREN (#(lambda (lambda (fold $0 $1 (lambda (lambda (cons $1 $0)))))) (cons RPAREN empty) $0)))":
                           "Enclose w/ parens",
                           "#(lambda (unfold $0 (lambda (empty? $0)) (lambda (car $0)) (lambda (#(lambda (lambda (fold $1 $1 (lambda (lambda (cdr (if (char-eq? $1 $2) $3 $0))))))) $0 SPACE))))":
                           "Abbreviate",
                           "#(lambda (lambda (fold $1 $1 (lambda (lambda (cdr (if (char-eq? $1 $2) $3 $0)))))))":
                           "Drop until char",
                           "#(lambda (lambda (fold $1 $1 (lambda (lambda (if (char-eq? $1 $2) empty (cons $1 $0)))))))":
                           "Take until char",
                           "#(lambda (lambda (#(lambda (lambda (fold $0 $1 (lambda (lambda (cons $1 $0)))))) (cons $0 $1))))":
                           "Append char",
                           "#(lambda (lambda (map (lambda (if (char-eq? $0 $1) $2 $0)))))":
                           "Substitute char",
                           "#(lambda (lambda (length (unfold $1 (lambda (char-eq? (car $0) $1)) (lambda ',') (lambda (cdr $0))))))":
                           "Index of char",
                           "#(lambda (#(lambda (lambda (fold $0 $1 (lambda (lambda (cons $1 $0)))))) $0 STRING))":
                           "Append const",
                           "#(lambda (lambda (fold $1 $1 (lambda (lambda (fold $0 $0 (lambda (lambda (cdr (if (char-eq? $1 $4) $0 (cons $1 $0)))))))))))":
                           "Last word",
                           "#(lambda (lambda (cons (car $1) (cons '.' (cons (car $0) (cons '.' empty))))))":
                           "Abbreviate name",
                           "#(lambda (lambda (cons (car $1) (cons $0 empty))))":
                           "First char+char",
                           "#(lambda (#(lambda (lambda (fold $0 $1 (lambda (lambda (cons $1 $0)))))) (#(lambda (lambda (fold $1 $1 (lambda (lambda (fold $0 $0 (lambda (lambda (cdr (if (char-eq? $1 $4) $0 (cons $1 $0))))))))))) STRING (index (length (cdr $0)) $0)) $0))":
                           "Ensure suffix"
                           
    }

    def makeGraph(ordering, fn):
        g = Digraph()
        g.graph_attr['rankdir'] = 'LR'

        for o in sorted(ordering.keys()):
            with g.subgraph(name='age%d'%o) as sg:
                sg.graph_attr['rank'] = 'same'
                for p in ordering[o]:
                    if str(p) in englishDescriptions:
                        thisLabel = '<<font face="boldfontname"><u>%s</u></font><br />%s>'%(englishDescriptions[str(p)],simplification[p])
                    else:
                        eprint("WARNING: Do not have an English description of:\n",p)
                        eprint()
                        thisLabel = "<%s>"%simplification[p]
                    sg.node(getName(p),
                            label=thisLabel)

            for p in ordering[o]:
                children = {k
                            for _,k in p.body.walk()
                            if k.isInvented}
                for k in children:
                    g.edge(name[k],name[p])

        try:
            g.render(fn,view=view)
            eprint("Exported primitive graph to",fn)
        except:
            eprint("Got some kind of error while trying to render primitive graph! Did you install graphviz/dot?")
        
        

    makeGraph(depth2primitives,prefix+'depth.pdf')
    makeGraph(age2primitives,prefix+'iter.pdf')
