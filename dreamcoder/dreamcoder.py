import datetime

import dill

from dreamcoder.compression import induceGrammar
from dreamcoder.recognition import *
from dreamcoder.enumeration import *
from dreamcoder.fragmentGrammar import *
from dreamcoder.taskBatcher import *
from dreamcoder.primitiveGraph import graphPrimitives
from dreamcoder.dreaming import backgroundHelmholtzEnumeration


class ECResult():
    def __init__(self, _=None,
                 frontiersOverTime=None,
                 testingSearchTime=None,
                 learningCurve=None,
                 grammars=None,
                 taskSolutions=None,
                 averageDescriptionLength=None,
                 parameters=None,
                 recognitionModel=None,
                 searchTimes=None,
                 recognitionTaskMetrics=None,
                 numTestingTasks=None,
                 sumMaxll=None,
                 testingSumMaxll=None,
                 hitsAtEachWake=None,
                 timesAtEachWake=None,
                 allFrontiers=None):
        self.frontiersOverTime = {} # Map from task to [frontier at iteration 1, frontier at iteration 2, ...]
        self.hitsAtEachWake = hitsAtEachWake or []
        self.timesAtEachWake = timesAtEachWake or []
        self.testingSearchTime = testingSearchTime or []
        self.searchTimes = searchTimes or []
        self.trainSearchTime = {} # map from task to search time
        self.testSearchTime = {} # map from task to search time
        self.recognitionTaskMetrics = recognitionTaskMetrics or {} 
        self.recognitionModel = recognitionModel
        self.averageDescriptionLength = averageDescriptionLength or []
        self.parameters = parameters
        self.learningCurve = learningCurve or []
        self.grammars = grammars or []
        self.taskSolutions = taskSolutions or {}
        self.numTestingTasks = numTestingTasks
        self.sumMaxll = sumMaxll or [] #TODO name change 
        self.testingSumMaxll = testingSumMaxll or [] #TODO name change
        self.allFrontiers = allFrontiers or {}

    def __repr__(self):
        attrs = ["{}={}".format(k, v) for k, v in self.__dict__.items()]
        return "ECResult({})".format(", ".join(attrs))

    def getTestingTasks(self):
        testing = []
        training = self.taskSolutions.keys()
        for t in self.recognitionTaskMetrics:
            if isinstance(t, Task) and t not in training: testing.append(t)
        return testing


    def recordFrontier(self, frontier):
        t = frontier.task
        if t not in self.frontiersOverTime: self.frontiersOverTime[t] = []
        self.frontiersOverTime[t].append(frontier)

    # Linux does not like files that have more than 256 characters
    # So when exporting the results we abbreviate the parameters
    abbreviations = {"frontierSize": "fs",
                     "useDSL": "DSL",
                     "taskReranker": "TRR",
                     "matrixRank": "MR",
                     "reuseRecognition": "RR",
                     "ensembleSize": "ES",
                     "recognitionTimeout": "RT",
                     "recognitionSteps": "RS",
                     "iterations": "it",
                     "maximumFrontier": "MF",
                     "pseudoCounts": "pc",
                     "auxiliaryLoss": "aux",
                     "structurePenalty": "L",
                     "helmholtzRatio": "HR",
                     "biasOptimal": "BO",
                     "contextual": "CO",
                     "topK": "K",
                     "enumerationTimeout": "ET",
                     "useRecognitionModel": "rec",
                     "use_ll_cutoff": "llcut",
                     "topk_use_only_likelihood": "topkNotMAP",
                     "activation": "act",
                     "storeTaskMetrics": 'STM',
                     "topkNotMAP": "tknm",
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
               useDSL=True,
               noConsolidation=False,
               mask=False,
               seed=0,
               addFullTaskMetrics=False,
               matrixRank=None,
               solver='ocaml',
               compressor="rust",
               biasOptimal=False,
               contextual=False,
               testingTasks=[],
               iterations=None,
               resume=None,
               enumerationTimeout=None,
               testingTimeout=None,
               testEvery=1,
               reuseRecognition=False,
               ensembleSize=1,
               useRecognitionModel=True,
               recognitionTimeout=None,
               recognitionSteps=None,
               helmholtzRatio=0.,
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
               outputPrefix=None,
               storeTaskMetrics=False,
               rewriteTaskMetrics=True,
               auxiliaryLoss=False,
               custom_wake_generative=None):
    if enumerationTimeout is None:
        eprint(
            "Please specify an enumeration timeout:",
            "explorationCompression(..., enumerationTimeout = ..., ...)")
        assert False
    if iterations is None:
        eprint(
            "Please specify a iteration count: explorationCompression(..., iterations = ...)")
        assert False
    if useRecognitionModel and featureExtractor is None:
        eprint("Warning: Recognition model needs feature extractor.",
               "Ignoring recognition model.")
        useRecognitionModel = False
    if ensembleSize > 1 and not useRecognitionModel:
        eprint("Warning: ensemble size requires using the recognition model, aborting.")
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
    if matrixRank is not None and not contextual:
        eprint("Matrix rank only applies to contextual recognition models, aborting")
        assert False
    assert useDSL or useRecognitionModel, "You specified that you didn't want to use the DSL AND you don't want to use the recognition model. Figure out what you want to use."
    if testingTimeout > 0 and len(testingTasks) == 0:
        eprint("You specified a testingTimeout, but did not provide any held out testing tasks, aborting.")
        assert False

    # We save the parameters that were passed into EC
    # This is for the purpose of exporting the results of the experiment
    parameters = {
        k: v for k,
        v in locals().items() if k not in {
            "tasks",
            "use_map_search_times",
            "seed",
            "activation",
            "grammar",
            "cuda",
            "_",
            "testingTimeout",
            "testEvery",
            "message",
            "CPUs",
            "outputPrefix",
            "resume",
            "resumeFrontierSize",
            "addFullTaskMetrics",
            "featureExtractor",
            "evaluationTimeout",
            "testingTasks",
            "compressor",
            "custom_wake_generative"} and v is not None}
    if not useRecognitionModel:
        for k in {"helmholtzRatio", "recognitionTimeout", "biasOptimal", "mask",
                  "contextual", "matrixRank", "reuseRecognition", "auxiliaryLoss", "ensembleSize"}:
            if k in parameters: del parameters[k]
    else: del parameters["useRecognitionModel"];
    if useRecognitionModel and not contextual:
        if "matrixRank" in parameters:
            del parameters["matrixRank"]
        if "mask" in parameters:
            del parameters["mask"]
    if not mask and 'mask' in parameters: del parameters["mask"]
    if not auxiliaryLoss and 'auxiliaryLoss' in parameters: del parameters['auxiliaryLoss']
    if not useDSL:
        for k in {"structurePenalty", "pseudoCounts", "aic"}:
            del parameters[k]
    else: del parameters["useDSL"]
    
    # Uses `parameters` to construct the checkpoint path
    def checkpointPath(iteration, extra=""):
        parameters["iterations"] = iteration
        kvs = [
            "{}={}".format(
                ECResult.abbreviate(k),
                parameters[k]) for k in sorted(
                parameters.keys())]
        return "{}_{}{}.pickle".format(outputPrefix, "_".join(kvs), extra)

    if message:
        message = " (" + message + ")"
    eprint("Running EC%s on %s @ %s with %d CPUs and parameters:" %
           (message, os.uname()[1], datetime.datetime.now(), CPUs))
    for k, v in parameters.items():
        eprint("\t", k, " = ", v)
    eprint("\t", "evaluationTimeout", " = ", evaluationTimeout)
    eprint("\t", "cuda", " = ", cuda)
    eprint()

    if addFullTaskMetrics:
        assert resume is not None, "--addFullTaskMetrics requires --resume"

    def reportMemory():
        eprint(f"Currently using this much memory: {getThisMemoryUsage()}")
    
    # Restore checkpoint
    if resume is not None:
        try:
            resume = int(resume)
            path = checkpointPath(resume)
        except ValueError:
            path = resume
        with open(path, "rb") as handle:
            result = dill.load(handle)
        resume = len(result.grammars) - 1
        eprint("Loaded checkpoint from", path)
        grammar = result.grammars[-1] if result.grammars else grammar
    else:  # Start from scratch
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


    # Set up the task batcher.
    if taskReranker == 'default':
        taskBatcher = DefaultTaskBatcher()
    elif taskReranker == 'random':
        taskBatcher = RandomTaskBatcher()
    elif taskReranker == 'randomShuffle':
        taskBatcher = RandomShuffleTaskBatcher(seed)
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

    # Check if we are just updating the full task metrics
    if addFullTaskMetrics:
        if testingTimeout is not None and testingTimeout > enumerationTimeout:
            enumerationTimeout = testingTimeout
        if result.recognitionModel is not None:
            _enumerator = lambda *args, **kw: result.recognitionModel.enumerateFrontiers(*args, **kw)
        else: _enumerator = lambda *args, **kw: multicoreEnumeration(result.grammars[-1], *args, **kw)
        enumerator = lambda *args, **kw: _enumerator(*args, 
                                                     maximumFrontier=maximumFrontier, 
                                                     CPUs=CPUs, evaluationTimeout=evaluationTimeout,
                                                     solver=solver,
                                                     **kw)
        trainFrontiers, _, trainingTimes = enumerator(tasks, enumerationTimeout=enumerationTimeout)
        testFrontiers, _, testingTimes = enumerator(testingTasks, enumerationTimeout=testingTimeout, testing=True)

        recognizer = result.recognitionModel
        updateTaskSummaryMetrics(result.recognitionTaskMetrics, trainingTimes, 'recognitionBestTimes')
        updateTaskSummaryMetrics(result.recognitionTaskMetrics, recognizer.taskGrammarLogProductions(tasks), 'taskLogProductions')
        updateTaskSummaryMetrics(result.recognitionTaskMetrics, recognizer.taskGrammarEntropies(tasks), 'taskGrammarEntropies')
        updateTaskSummaryMetrics(result.recognitionTaskMetrics, result.recognitionModel.taskAuxiliaryLossLayer(tasks), 'taskAuxiliaryLossLayer')
        
        updateTaskSummaryMetrics(result.recognitionTaskMetrics, testingTimes, 'heldoutTestingTimes')
        updateTaskSummaryMetrics(result.recognitionTaskMetrics, recognizer.taskGrammarLogProductions(testingTasks), 'heldoutTaskLogProductions')
        updateTaskSummaryMetrics(result.recognitionTaskMetrics, recognizer.taskGrammarEntropies(testingTasks), 'heldoutTaskGrammarEntropies')
        updateTaskSummaryMetrics(result.recognitionTaskMetrics, result.recognitionModel.taskAuxiliaryLossLayer(testingTasks), 'heldoutAuxiliaryLossLayer')

        updateTaskSummaryMetrics(result.recognitionTaskMetrics, {f.task: f
                                                                 for f in trainFrontiers + testFrontiers
                                                                 if len(f) > 0},
                                 'frontier')
        SUFFIX = ".pickle"
        assert path.endswith(SUFFIX)
        path = path[:-len(SUFFIX)] + "_FTM=True" + SUFFIX
        with open(path, "wb") as handle: dill.dump(result, handle)
        if useRecognitionModel: ECResult.clearRecognitionModel(path)
            
        sys.exit(0)
    
    
    for j in range(resume or 0, iterations):
        if storeTaskMetrics and rewriteTaskMetrics:
            eprint("Resetting task metrics for next iteration.")
            result.recognitionTaskMetrics = {}

        reportMemory()

        # Evaluate on held out tasks if we have them
        if testingTimeout > 0 and ((j % testEvery == 0) or (j == iterations - 1)):
            eprint("Evaluating on held out testing tasks for iteration: %d" % (j))
            evaluateOnTestingTasks(result, testingTasks, grammar,
                                   CPUs=CPUs, maximumFrontier=maximumFrontier,
                                   solver=solver,
                                   enumerationTimeout=testingTimeout, evaluationTimeout=evaluationTimeout)            
        # If we have to also enumerate Helmholtz frontiers,
        # do this extra sneaky in the background
        if useRecognitionModel and biasOptimal and helmholtzRatio > 0 and \
           all( str(p) != "REAL" for p in grammar.primitives ): # real numbers don't support this
            # the DSL is fixed, so the dreams are also fixed. don't recompute them.
            if useDSL or 'helmholtzFrontiers' not in locals():
                helmholtzFrontiers = backgroundHelmholtzEnumeration(tasks, grammar, enumerationTimeout,
                                                                    evaluationTimeout=evaluationTimeout,
                                                                    special=featureExtractor.special)
            else:
                print("Reusing dreams from previous iteration.")
        else:
            helmholtzFrontiers = lambda: []

        reportMemory()

        # Get waking task batch.
        wakingTaskBatch = taskBatcher.getTaskBatch(result, tasks, taskBatchSize, j)
        eprint("Using a waking task batch of size: " + str(len(wakingTaskBatch)))

        # WAKING UP
        if useDSL:
            wake_generative = custom_wake_generative if custom_wake_generative is not None else default_wake_generative
            topDownFrontiers, times = wake_generative(grammar, wakingTaskBatch,
                                                      solver=solver,
                                                      maximumFrontier=maximumFrontier,
                                                      enumerationTimeout=enumerationTimeout,
                                                      CPUs=CPUs,
                                                      evaluationTimeout=evaluationTimeout)
            result.trainSearchTime = {t: tm for t, tm in times.items() if tm is not None}
        else:
            eprint("Skipping top-down enumeration because we are not using the generative model")
            topDownFrontiers, times = [], {t: None for t in wakingTaskBatch }

        tasksHitTopDown = {f.task for f in topDownFrontiers if not f.empty}
        result.hitsAtEachWake.append(len(tasksHitTopDown))

        reportMemory()

        # Combine topDownFrontiers from this task batch with all frontiers.
        for f in topDownFrontiers:
            if f.task not in result.allFrontiers: continue # backward compatibility with old checkpoints
            result.allFrontiers[f.task] = result.allFrontiers[f.task].combine(f).topK(maximumFrontier)

        eprint("Frontiers discovered top down: " + str(len(tasksHitTopDown)))
        eprint("Total frontiers: " + str(len([f for f in result.allFrontiers.values() if not f.empty])))

        # Train + use recognition model
        if useRecognitionModel:
            # Should we initialize the weights to be what they were before?
            previousRecognitionModel = None
            if reuseRecognition and result.recognitionModel is not None:
                previousRecognitionModel = result.recognitionModel

            thisRatio = helmholtzRatio
            if j == 0 and not biasOptimal: thisRatio = 0
            if all( f.empty for f in result.allFrontiers.values() ): thisRatio = 1.                

            tasksHitBottomUp = \
             sleep_recognition(result, grammar, wakingTaskBatch, tasks, testingTasks, result.allFrontiers.values(),
                               ensembleSize=ensembleSize, featureExtractor=featureExtractor, mask=mask,
                               activation=activation, contextual=contextual, biasOptimal=biasOptimal,
                               previousRecognitionModel=previousRecognitionModel, matrixRank=matrixRank,
                               timeout=recognitionTimeout, evaluationTimeout=evaluationTimeout,
                               enumerationTimeout=enumerationTimeout,
                               helmholtzRatio=thisRatio, helmholtzFrontiers=helmholtzFrontiers(),
                               auxiliaryLoss=auxiliaryLoss, cuda=cuda, CPUs=CPUs, solver=solver,
                               recognitionSteps=recognitionSteps, maximumFrontier=maximumFrontier)

            showHitMatrix(tasksHitTopDown, tasksHitBottomUp, wakingTaskBatch)
            
        # Record the new topK solutions
        result.taskSolutions = {f.task: f.topK(topK)
                                for f in result.allFrontiers.values()}
        for f in result.allFrontiers.values(): result.recordFrontier(f)
        result.learningCurve += [
            sum(f is not None and not f.empty for f in result.taskSolutions.values())]
        updateTaskSummaryMetrics(result.recognitionTaskMetrics, {f.task: f
                                                                 for f in result.allFrontiers.values()
                                                                 if len(f) > 0},
                                 'frontier')                
        
        # Sleep-G
        if useDSL and not(noConsolidation):
            eprint(f"Currently using this much memory: {getThisMemoryUsage()}")
            grammar = consolidate(result, grammar, topK=topK, pseudoCounts=pseudoCounts, arity=arity, aic=aic,
                                  structurePenalty=structurePenalty, compressor=compressor, CPUs=CPUs,
                                  iteration=j)
            eprint(f"Currently using this much memory: {getThisMemoryUsage()}")
        else:
            eprint("Skipping consolidation.")
            result.grammars.append(grammar)
            
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

def evaluateOnTestingTasks(result, testingTasks, grammar, _=None,
                           CPUs=None, solver=None, maximumFrontier=None, enumerationTimeout=None, evaluationTimeout=None):
    if result.recognitionModel is not None:
        recognizer = result.recognitionModel
        testingFrontiers, times = \
         recognizer.enumerateFrontiers(testingTasks, 
                                       CPUs=CPUs,
                                       solver=solver,
                                       maximumFrontier=maximumFrontier,
                                       enumerationTimeout=enumerationTimeout,
                                       evaluationTimeout=evaluationTimeout,
                                       testing=True)
        updateTaskSummaryMetrics(result.recognitionTaskMetrics, recognizer.taskGrammarLogProductions(testingTasks), 'heldoutTaskLogProductions')
        updateTaskSummaryMetrics(result.recognitionTaskMetrics, recognizer.taskGrammarEntropies(testingTasks), 'heldoutTaskGrammarEntropies')
        updateTaskSummaryMetrics(result.recognitionTaskMetrics, recognizer.taskGrammarEntropies(testingTasks), 'heldoutTaskGrammarEntropies')
    else:
        testingFrontiers, times = multicoreEnumeration(grammar, testingTasks, 
                                                       solver=solver,
                                                       maximumFrontier=maximumFrontier,
                                                       enumerationTimeout=enumerationTimeout,
                                                       CPUs=CPUs,
                                                       evaluationTimeout=evaluationTimeout,
                                                       testing=True)
    updateTaskSummaryMetrics(result.recognitionTaskMetrics, times, 'heldoutTestingTimes')
    updateTaskSummaryMetrics(result.recognitionTaskMetrics,
                                     {f.task: f for f in testingFrontiers if len(f) > 0 },
                                     'frontier')
    for f in testingFrontiers: result.recordFrontier(f)
    result.testSearchTime = {t: tm for t, tm in times.items() if tm is not None}
    times = [t for t in times.values() if t is not None ]
    eprint("\n".join(f.summarize() for f in testingFrontiers))
    summaryStatistics("Testing tasks", times)
    eprint("Hits %d/%d testing tasks" % (len(times), len(testingTasks)))
    result.testingSearchTime.append(times)

        
def default_wake_generative(grammar, tasks, 
                    maximumFrontier=None,
                    enumerationTimeout=None,
                    CPUs=None,
                    solver=None,
                    evaluationTimeout=None):
    topDownFrontiers, times = multicoreEnumeration(grammar, tasks, 
                                                   maximumFrontier=maximumFrontier,
                                                   enumerationTimeout=enumerationTimeout,
                                                   CPUs=CPUs,
                                                   solver=solver,
                                                   evaluationTimeout=evaluationTimeout)
    eprint("Generative model enumeration results:")
    eprint(Frontier.describe(topDownFrontiers))
    summaryStatistics("Generative model", [t for t in times.values() if t is not None])
    return topDownFrontiers, times

def sleep_recognition(result, grammar, taskBatch, tasks, testingTasks, allFrontiers, _=None,
                      ensembleSize=1, featureExtractor=None, matrixRank=None, mask=False,
                      activation=None, contextual=True, biasOptimal=True,
                      previousRecognitionModel=None, recognitionSteps=None,
                      timeout=None, enumerationTimeout=None, evaluationTimeout=None,
                      helmholtzRatio=None, helmholtzFrontiers=None, maximumFrontier=None,
                      auxiliaryLoss=None, cuda=None, CPUs=None, solver=None):
    eprint("Using an ensemble size of %d. Note that we will only store and test on the best recognition model." % ensembleSize)

    featureExtractorObjects = [featureExtractor(tasks, testingTasks=testingTasks, cuda=cuda) for i in range(ensembleSize)]
    recognizers = [RecognitionModel(featureExtractorObjects[i],
                                    grammar,
                                    mask=mask,
                                    rank=matrixRank,
                                    activation=activation,
                                    cuda=cuda,
                                    contextual=contextual,
                                    previousRecognitionModel=previousRecognitionModel,
                                    id=i) for i in range(ensembleSize)]
    eprint(f"Currently using this much memory: {getThisMemoryUsage()}")
    trainedRecognizers = parallelMap(min(CPUs,len(recognizers)),
                                     lambda recognizer: recognizer.train(allFrontiers,
                                                                         biasOptimal=biasOptimal,
                                                                         helmholtzFrontiers=helmholtzFrontiers, 
                                                                         CPUs=CPUs,
                                                                         evaluationTimeout=evaluationTimeout,
                                                                         timeout=timeout,
                                                                         steps=recognitionSteps,
                                                                         helmholtzRatio=helmholtzRatio,
                                                                         auxLoss=auxiliaryLoss,
                                                                         vectorized=True),
                                     recognizers,
                                     seedRandom=True)
    eprint(f"Currently using this much memory: {getThisMemoryUsage()}")
    # Enumerate frontiers for each of the recognizers.
    eprint("Trained an ensemble of %d recognition models, now enumerating." % len(trainedRecognizers))
    ensembleFrontiers, ensembleTimes, ensembleRecognitionTimes = [], [], []
    mostTasks = 0
    bestRecognizer = None
    totalTasksHitBottomUp = set()
    for recIndex, recognizer in enumerate(trainedRecognizers):
        eprint("Enumerating from recognizer %d of %d" % (recIndex, len(trainedRecognizers)))
        bottomupFrontiers, allRecognitionTimes = \
                        recognizer.enumerateFrontiers(taskBatch, 
                                                      CPUs=CPUs,
                                                      maximumFrontier=maximumFrontier,
                                                      enumerationTimeout=enumerationTimeout,
                                                      evaluationTimeout=evaluationTimeout,
                                                      solver=solver)
        ensembleFrontiers.append(bottomupFrontiers)
        ensembleTimes.append([t for t in allRecognitionTimes.values() if t is not None])
        ensembleRecognitionTimes.append(allRecognitionTimes)

        recognizerTasksHitBottomUp = {f.task for f in bottomupFrontiers if not f.empty}
        totalTasksHitBottomUp.update(recognizerTasksHitBottomUp)
        eprint("Recognizer %d solved %d/%d tasks; total tasks solved is now %d." % (recIndex, len(recognizerTasksHitBottomUp), len(tasks), len(totalTasksHitBottomUp)))
        if len(recognizerTasksHitBottomUp) >= mostTasks:
            # TODO (cathywong): could consider keeping the one that put the highest likelihood on the solved tasks.
            bestRecognizer = recIndex

    # Store the recognizer that discovers the most frontiers in the result.
    eprint("Best recognizer: %d." % bestRecognizer)
    result.recognitionModel = trainedRecognizers[bestRecognizer]
    result.trainSearchTime = {tk: tm for tk, tm in ensembleRecognitionTimes[bestRecognizer].items()
                              if tm is not None}
    updateTaskSummaryMetrics(result.recognitionTaskMetrics, ensembleRecognitionTimes[bestRecognizer], 'recognitionBestTimes')
    updateTaskSummaryMetrics(result.recognitionTaskMetrics, result.recognitionModel.taskHiddenStates(tasks), 'hiddenState')
    updateTaskSummaryMetrics(result.recognitionTaskMetrics, result.recognitionModel.taskGrammarLogProductions(tasks), 'taskLogProductions')
    updateTaskSummaryMetrics(result.recognitionTaskMetrics, result.recognitionModel.taskGrammarEntropies(tasks), 'taskGrammarEntropies')
    if contextual:
        updateTaskSummaryMetrics(result.recognitionTaskMetrics,
                                 result.recognitionModel.taskGrammarStartProductions(tasks),
                                 'startProductions')

    result.hitsAtEachWake.append(len(totalTasksHitBottomUp))
    eprint(f"Currently using this much memory: {getThisMemoryUsage()}")

    """ Rescore and combine the frontiers across the ensemble of recognition models."""
    eprint("Recognition model enumeration results for the best recognizer.")
    eprint(Frontier.describe(ensembleFrontiers[bestRecognizer]))
    summaryStatistics("Recognition model", ensembleTimes[bestRecognizer])

    eprint("Cumulative results for the full ensemble of %d recognizers: " % len(trainedRecognizers))
    # Rescore all of the ensemble frontiers according to the generative model
    # and then combine w/ original frontiers
    for bottomupFrontiers in ensembleFrontiers:
        for b in bottomupFrontiers:
            if b.task not in result.allFrontiers: continue # backwards compatibility with old checkpoints
            result.allFrontiers[b.task] = result.allFrontiers[b.task].\
                                          combine(grammar.rescoreFrontier(b)).\
                                          topK(maximumFrontier)

    eprint("Frontiers discovered bottom up: " + str(len(totalTasksHitBottomUp)))
    eprint("Total frontiers: " + str(len([f for f in result.allFrontiers.values() if not f.empty])))

    result.searchTimes.append(ensembleTimes[bestRecognizer])
    if len(ensembleTimes[bestRecognizer]) > 0:
        eprint("Average search time: ", int(mean(ensembleTimes[bestRecognizer]) + 0.5),
               "sec.\tmedian:", int(median(ensembleTimes[bestRecognizer]) + 0.5),
               "\tmax:", int(max(ensembleTimes[bestRecognizer]) + 0.5),
               "\tstandard deviation", int(standardDeviation(ensembleTimes[bestRecognizer]) + 0.5))
    return totalTasksHitBottomUp

def consolidate(result, grammar, _=None, topK=None, arity=None, pseudoCounts=None, aic=None,
                structurePenalty=None, compressor=None, CPUs=None, iteration=None):
    eprint("Showing the top 5 programs in each frontier being sent to the compressor:")
    for f in result.allFrontiers.values():
        if f.empty:
            continue
        eprint(f.task)
        for e in f.normalize().topK(5):
            eprint("%.02f\t%s" % (e.logPosterior, e.program))
        eprint()

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
                                                      topk_use_only_likelihood=False,
                                                      backend=compressor, CPUs=CPUs, iteration=iteration)
        # Store compression frontiers in the result.
        for c in compressionFrontiers:
            result.allFrontiers[c.task] = c.topK(0) if c in needToSupervise else c


    result.grammars.append(grammar)
    eprint("Grammar after iteration %d:" % (iteration + 1))
    eprint(grammar)
    
    return grammar
    


def commandlineArguments(_=None,
                         iterations=None,
                         enumerationTimeout=None,
                         testEvery=1,
                         topK=1,
                         reuseRecognition=False,
                         CPUs=1,
                         solver='ocaml',
                         compressor="ocaml",
                         useRecognitionModel=True,
                         recognitionTimeout=None,
                         activation='relu',
                         helmholtzRatio=1.,
                         featureExtractor=None,
                         cuda=None,
                         maximumFrontier=None,
                         pseudoCounts=1.0, aic=1.0,
                         structurePenalty=0.001, a=0,
                         taskBatchSize=None, taskReranker="default",
                         extras=None,
                         storeTaskMetrics=False,
                        rewriteTaskMetrics=True):
    if cuda is None:
        cuda = torch.cuda.is_available()
    print("CUDA is available?:", torch.cuda.is_available())
    print("using cuda?:", cuda)
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
    parser.add_argument("-t", "--enumerationTimeout",
                        default=enumerationTimeout,
                        help="In seconds. default: %s" % enumerationTimeout,
                        type=int)
    parser.add_argument("-R", "--recognitionTimeout",
                        default=recognitionTimeout,
                        help="In seconds. Amount of time to train the recognition model on each iteration. Defaults to enumeration timeout.",
                        type=int)
    parser.add_argument("-RS", "--recognitionSteps",
                        default=None,
                        help="Number of gradient steps to train the recognition model. Can be specified instead of train time.",
                        type=int)
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
    parser.add_argument("--recognition",
                        dest="useRecognitionModel",
                        action="store_true",
                        help="""Enable bottom-up neural recognition model.
                        Default: %s""" % useRecognitionModel)
    parser.add_argument("--ensembleSize",
                        dest="ensembleSize",
                        default=1,
                        help="Number of recognition models to train and enumerate from at each iteration.",
                        type=int)
    parser.add_argument("-g", "--no-recognition",
                        dest="useRecognitionModel",
                        action="store_false",
                        help="""Disable bottom-up neural recognition model.
                        Default: %s""" % (not useRecognitionModel))
    parser.add_argument("-d", "--no-dsl",
                        dest="useDSL",
                        action="store_false",
                        help="""Disable DSL enumeration and updating.""")
    parser.add_argument("--no-consolidation",
                        dest="noConsolidation",
                        action="store_true",
                        help="""Disable DSL updating.""")
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
        "--seed",
        type=int,
        default=0,
        help="Random seed. Currently this only matters for random batching strategies.")
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
        "--solver",
        choices=[
            "ocaml",
            "pypy",
            "python"],
        default=solver,
        help="""Solver for enumeration.
                        Default: %s""" %
        solver)
    parser.add_argument(
        "-r",
        "--Helmholtz",
        dest="helmholtzRatio",
        help="""When training recognition models, what fraction of the training data should be samples from the generative model? Default %f""" %
        helmholtzRatio,
        default=helmholtzRatio,
        type=float)
    parser.add_argument(
        "--compressor",
        default=compressor,
        choices=["pypy","rust","vs","pypy_vs","ocaml","memorize"])
    parser.add_argument(
        "--matrixRank",
        help="Maximum rank of bigram transition matrix for contextual recognition model. Defaults to full rank.",
        default=None,
        type=int)
    parser.add_argument(
        "--mask",
        help="Unconditional bigram masking",
        default=False, action="store_true")
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
                        nargs='+',
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
        default=True,
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
                        nargs='+',
                        type=str)
    parser.add_argument("--auxiliary",
                        action="store_true", default=False,
                        help="Add auxiliary classification loss to recognition network training",
                        dest="auxiliaryLoss")
    parser.add_argument("--addFullTaskMetrics",
                        help="Only to be used in conjunction with --resume. Loads checkpoint, solves both testing and training tasks, stores frontiers, solve times, and task metrics, and then dies.",
                        default=False,
                        action="store_true")
    parser.add_argument("--countParameters",
                        help="Load a checkpoint then report how many parameters are in the recognition model.",
                        default=None, type=str)
    parser.set_defaults(useRecognitionModel=useRecognitionModel,
                        useDSL=True,
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
        
        for n,pg in enumerate(v["primitive-graph"]):
            with open(pg,'rb') as handle:
                result = dill.load(handle)
            graphPrimitives(result,f"figures/deepProgramLearning/{sys.argv[0]}{n}",view=True)
        sys.exit(0)
    else:
        del v["primitive-graph"]

    if v["addTaskMetrics"] is not None:
        for path in v["addTaskMetrics"]:
            with open(path,'rb') as handle:
                result = dill.load(handle)
            addTaskMetrics(result, path)
        sys.exit(0)
    else:
        del v["addTaskMetrics"]

    if v["useRecognitionModel"] and v["recognitionTimeout"] is None:
        v["recognitionTimeout"] = v["enumerationTimeout"]

    if v["countParameters"]:
        with open(v["countParameters"], "rb") as handle:
            result = dill.load(handle)
        eprint("The recognition model has",
               sum(p.numel() for p in result.recognitionModel.parameters() if p.requires_grad),
               "trainable parameters and",
               sum(p.numel() for p in result.recognitionModel.parameters() ),
               "total parameters.\n",
               "The feature extractor accounts for",
               sum(p.numel() for p in result.recognitionModel.featureExtractor.parameters() ),
               "of those parameters.\n",
               "The grammar builder accounts for",
               sum(p.numel() for p in result.recognitionModel.grammarBuilder.parameters() ),
               "of those parameters.\n")
        sys.exit(0)
    del v["countParameters"]
        
        
    return v

def addTaskMetrics(result, path):
    """Adds a task metrics to ECResults that were pickled without them."""
    with torch.no_grad(): return addTaskMetrics_(result, path)
def addTaskMetrics_(result, path):
    SUFFIX = '.pickle'
    assert path.endswith(SUFFIX)

    tasks = result.taskSolutions.keys()
    everyTask = set(tasks)
    for t in result.recognitionTaskMetrics:
        if isinstance(t, Task) and t not in everyTask: everyTask.add(t)

    eprint(f"Found {len(tasks)} training tasks.")
    eprint(f"Scrounged up {len(everyTask) - len(tasks)} testing tasks.")
    if not hasattr(result, "recognitionTaskMetrics") or result.recognitionTaskMetrics is None:
        result.recognitionTaskMetrics = {}

    # If task has images, store them.
    if hasattr(list(tasks)[0], 'getImage'):
        images = {t: t.getImage(pretty=True) for t in tasks}
        updateTaskSummaryMetrics(result.recognitionTaskMetrics, images, 'taskImages')

    if hasattr(list(tasks)[0], 'highresolution'):
        images = {t: t.highresolution for t in tasks}
        updateTaskSummaryMetrics(result.recognitionTaskMetrics, images, 'taskImages')

    updateTaskSummaryMetrics(result.recognitionTaskMetrics, result.recognitionModel.auxiliaryPrimitiveEmbeddings(), 'auxiliaryPrimitiveEmbeddings')
    updateTaskSummaryMetrics(result.recognitionTaskMetrics, result.recognitionModel.taskAuxiliaryLossLayer(tasks), 'taskAuxiliaryLossLayer')
    updateTaskSummaryMetrics(result.recognitionTaskMetrics, result.recognitionModel.taskAuxiliaryLossLayer(everyTask), 'every_auxiliaryLossLayer')

    updateTaskSummaryMetrics(result.recognitionTaskMetrics, result.recognitionModel.taskGrammarFeatureLogProductions(tasks), 'grammarFeatureLogProductions')
    updateTaskSummaryMetrics(result.recognitionTaskMetrics, result.recognitionModel.taskGrammarFeatureLogProductions(everyTask), 'every_grammarFeatureLogProductions')
    updateTaskSummaryMetrics(result.recognitionTaskMetrics, result.recognitionModel.taskGrammarLogProductions(tasks), 'contextualLogProductions')
    updateTaskSummaryMetrics(result.recognitionTaskMetrics, result.recognitionModel.taskGrammarLogProductions(everyTask), 'every_contextualLogProductions')
    updateTaskSummaryMetrics(result.recognitionTaskMetrics, result.recognitionModel.taskHiddenStates(tasks), 'hiddenState')
    updateTaskSummaryMetrics(result.recognitionTaskMetrics, result.recognitionModel.taskHiddenStates(everyTask), 'every_hiddenState')
    g = result.grammars[-2] # the final entry in result.grammars is a grammar that we have not used yet
    updateTaskSummaryMetrics(result.recognitionTaskMetrics, {f.task: f.expectedProductionUses(g)
                                                             for f in result.taskSolutions.values()
                                                             if len(f) > 0},
                             'expectedProductionUses')
    updateTaskSummaryMetrics(result.recognitionTaskMetrics, {f.task: f.expectedProductionUses(g)
                                                             for t, metrics in result.recognitionTaskMetrics.items()
                                                             if "frontier" in metrics
                                                             for f in [metrics["frontier"]] 
                                                             if len(f) > 0},
                             'every_expectedProductionUses')
    if False:
        eprint(f"About to do an expensive Monte Carlo simulation w/ {len(tasks)} tasks")
        updateTaskSummaryMetrics(result.recognitionTaskMetrics,
                                 {task: result.recognitionModel.grammarOfTask(task).untorch().expectedUsesMonteCarlo(task.request, debug=False)
                                  for task in tasks },
                                 'expectedProductionUsesMonteCarlo')
    try:
        updateTaskSummaryMetrics(result.recognitionTaskMetrics,
                                 result.recognitionModel.taskGrammarStartProductions(tasks),
                                 'startProductions')
    except: pass # can fail if we do not have a contextual model

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

