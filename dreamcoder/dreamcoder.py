import datetime

import dill
import gc

from dreamcoder.compression import induceGrammar
from dreamcoder.recognition import *
from dreamcoder.enumeration import *
from dreamcoder.fragmentGrammar import *
from dreamcoder.taskBatcher import *
from dreamcoder.primitiveGraph import graphPrimitives
from dreamcoder.dreaming import backgroundHelmholtzEnumeration
from dreamcoder.parser import *
from dreamcoder.languageUtilities import *
from dreamcoder.translation import *

class ECResult():
    def __init__(self, _=None,
                 frontiersOverTime=None,
                 testingSearchTime=None,
                 learningCurve=None,
                 grammars=None,
                 taskSolutions=None,
                 averageDescriptionLength=None,
                 parameters=None,
                 models=None,
                 recognitionModel=None,
                 searchTimes=None,
                 recognitionTaskMetrics=None,
                 numTestingTasks=None,
                 sumMaxll=None,
                 testingSumMaxll=None,
                 hitsAtEachWake=None,
                 timesAtEachWake=None,
                 allFrontiers=None,
                 taskLanguage=None,
                 tasksAttempted=None,):
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
        self.taskLanguage = taskLanguage or {} # Maps from task names to language.
        self.models = models or [] # List of recognition models.
        self.tasksAttempted = tasksAttempted or set() # Tasks we have attempted so far.

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
                     "matrixRank": "MR",
                     "reuseRecognition": "RR",
                     "ensembleSize": "ES",
                     "recognitionTimeout": "RT",
                     "recognitionSteps": "RS",
                     "recognitionEpochs": "RE",
                     'useWakeLanguage' : "LANG",
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
                     "recognition_0": "rec",
                     "use_ll_cutoff": "llcut",
                     "topk_use_only_likelihood": "topLL",
                     "activation": "act",
                     "storeTaskMetrics": 'STM',
                     "topkNotMAP": "tknm",
                     "rewriteTaskMetrics": "RW",
                     'taskBatchSize': 'batch',
                     'language_encoder' : 'lang_ft',
                     'noConsolidation': 'no_dsl'}

    @staticmethod
    def abbreviate(parameter): return ECResult.abbreviations.get(parameter, parameter)
    
    @staticmethod
    def abbreviate_value(value): 
        if type(value) == bool:
            return str(value)[0]
        else:
            return value
            
    @staticmethod
    def parameterOfAbbreviation(abbreviation):
        return ECResult.abbreviationToParameter.get(abbreviation, abbreviation)

    @staticmethod
    def clearRecognitionModel(path):
        SUFFIX = '.pickle'
        assert path.endswith(SUFFIX)
        
        with open(path,'rb') as handle:
            result = dill.load(handle)
        
        result.models = []
        result.recognitionModel = None
        result.parser = None
        
        clearedPath = path[:-len(SUFFIX)] + "_graph=True" + SUFFIX
        with open(clearedPath,'wb') as handle:
            result = dill.dump(result, handle)
        eprint(" [+] Cleared recognition models from:")
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
               compressor="ocaml",
               biasOptimal=True,
               contextual=True,
               testingTasks=[],
               iterations=None,
               resume=None,
               initialTimeout=None,
               initialTimeoutIterations=None,
               enumerationTimeout=None,
               testingTimeout=None,
               testEvery=1,
               skip_first_test=False,
               test_only_after_recognition=False,
               test_dsl_only=False,   
               reuseRecognition=False,
               ensembleSize=1,
               # Recognition parameters.
               recognition_0=["examples"],
               recognition_1=[],
               # SMT parameters.
               moses_dir=None,
               smt_phrase_length=None,
               pretrained_word_embeddings=False,
               smt_pseudoalignments=0,
               finetune_1=False,
               helmholtz_nearest_language=0,
               language_encoder=None,
               featureExtractor=None,
               languageDataset=None,
               condition_independently_on_language_descriptions=False,
               recognitionEpochs=None,
               recognitionTimeout=None,
               recognitionSteps=None,
               helmholtzRatio=0.,
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
               custom_wake_generative=None,
               interactive=False,
               parser=None,
               interactiveTasks=None,
               taskDataset=None,
               languageDatasetDir=None,
               useWakeLanguage=False,
               debug=False,
               synchronous_grammar=False,
               language_compression=False,
               lc_score=False,
               max_compression=0,
               max_mem_per_enumeration_thread=1000000,
               # Entrypoint flags for integration tests. If these are set, we return early at semantic breakpoints in the iteration.
               test_task_language=False, # Integration test on the language we add to tasks.
               test_background_helmholtz=False, # Integration test for enumerating Helmholtz frontiers in the background.
               test_wake_generative_enumeration=False, # Integration test for enumeration.
               test_sleep_recognition_0=False, # Integration test for the examples-only recognizer.
               test_sleep_recognition_1=False, # Integration test for the language-based recognizer.
               test_next_iteration_settings=False, # Integration test for the second iteration.
               ):
    if enumerationTimeout is None:
        eprint(
            "Please specify an enumeration timeout:",
            "explorationCompression(..., enumerationTimeout = ..., ...)")
        assert False
    if iterations is None:
        eprint(
            "Please specify a iteration count: explorationCompression(..., iterations = ...)")
        assert False
    if (("examples" in recognition_0) or ("examples" in recognition_1))  and featureExtractor is None:
        eprint("Warning: Recognition models need examples feature extractor, but none found")
        assert False
    if (("language" in recognition_0) or ("language" in recognition_1))  and language_encoder is None:
        eprint("Warning: Recognition models need language encoder, but none found")
        assert False
    if matrixRank is not None and not contextual:
        eprint("Matrix rank only applies to contextual recognition models, aborting")
        assert False

    if testingTimeout > 0 and len(testingTasks) == 0:
        eprint("You specified a testingTimeout, but did not provide any held out testing tasks, aborting.")
        assert False
    
    model_inputs = [recognition_0, recognition_1]
    n_models = len([m for m in model_inputs if len(m) > 0])
    if len(recognitionEpochs) == 1 and len(recognitionEpochs) < len(model_inputs):
        recognitionEpochs = recognitionEpochs * len(model_inputs)
    def print_recognition_model_summary():
        eprint("-------------------Recognition Model Summary-------------------")
        eprint(f"Found n=[{n_models}] recognition models.")
        for i, model in enumerate(model_inputs):
            if len(model) < 1: continue
            eprint(f"Model {i}: {model}")
            if "examples" in model:
                eprint(f"Examples encoder: {str(featureExtractor.__name__)}")
            if "language" in model:
                eprint(f"Language encoder: {language_encoder}")
                eprint(f"Language dataset or datasets: {languageDataset}")
            if recognitionEpochs is not None:
                eprint(f"Epochs: [{recognitionEpochs[i]}]; contextual: {contextual}")
            elif recognitionSteps is not None:
                eprint(f"Steps: [{recognitionSteps}]; contextual: {contextual}")
            elif recognitionTimeout is not None:
                eprint(f"Timeout: [{recognitionTimeout}]; contextual: {contextual}")
            if synchronous_grammar:
                eprint(f"Incuding synchronous grammar to train recognition model with language.")
            if language_compression:
                eprint(f"Using a synchronous grammar during compression.")
            if i > 0:
                eprint(f"Use n={helmholtz_nearest_language} nearest language examples for Helmholtz")
                eprint(f"Finetune from examples: {finetune_1}")
            print("\n")
                
        eprint("--------------------------------------")
    print_recognition_model_summary()

    # TODO: figure out how to save the parameters without removing them here.
    # We save the parameters that were passed into EC
    # This is for the purpose of exporting the results of the experiment
    parameters = {
        k: v for k,
        v in locals().items() if k not in {
            "taskReranker",
            "languageDatasetDir",
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
            "custom_wake_generative",
            "interactive",
            "interactiveTasks",
            "parser",
            "print_recognition_model_summary",
            "condition_independently_on_language_descriptions",
            "solver"} and v is not None}
    if not recognition_0:
        for k in {"helmholtzRatio", "recognitionTimeout", "biasOptimal", "mask",
                  "contextual", "matrixRank", "reuseRecognition", "auxiliaryLoss", "ensembleSize"}:
            if k in parameters: del parameters[k]
    else: del parameters["recognition_0"];
    if recognition_0 and not contextual:
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
    
    if languageDataset:
        parameters["languageDataset"] = ",".join([os.path.basename(d).split(".")[0] for d in languageDataset])
    
    
    # Uses `parameters` to construct the checkpoint path
    def checkpointPath(iteration, extra=""):
        # Exclude from path, but not from saving in parameters.
        exclude_from_path = [
            "model_inputs",
            "language_encoder",
            "recognition_0",
            "recognition_1",
            "helmholtz_nearest_language",
            "taskDataset",
            "finetune_1",
            "recognitionEpochs",
            "languageDataset",
            "moses_dir",
            "debug",
            "smt_phrase_length",
            "pretrained_word_embeddings",
            "smt_pseudoalignments",
            "synchronous_grammar",
            "language_compression",
            "lc_score",
            "max_compression",
            "skip_first_test",
            "test_only_after_recognition",
            "n_models",
            "test_dsl_only",
            "initialTimeout",
            "initialTimeoutIterations"
        ]
        parameters["iterations"] = iteration
        checkpoint_params = [k for k in sorted(parameters.keys()) if k not in exclude_from_path and not k.startswith('test_')]
        kvs = [
            "{}={}".format(
                ECResult.abbreviate(k),
                ECResult.abbreviate_value(parameters[k])) for k in checkpoint_params]
        return "{}_{}{}.pickle".format(outputPrefix, "_".join(kvs), extra)
    
    print(f"Checkpoints will be written to [{checkpointPath('iter')}]")
    print(f"Checkpoint path len = [{len(checkpointPath('iter'))}]")
    assert(len(checkpointPath('iter')) < 256)
    
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
        # Backward compatability if we weren't tracking attempted tasks.
        if not hasattr(result, 'tasksAttempted'): result.tasksAttempted = set()
    
        # Use any new tasks.
        numTestingTasks = len(testingTasks) if len(testingTasks) != 0 else None
        result.numTestingTasks = numTestingTasks
        
        new_tasks = [t for t in tasks if t not in result.allFrontiers]
        new_testing = [t for t in testingTasks if t.name not in result.taskLanguage]
        print(f"Found {len(new_tasks)} new tasks and {len(new_testing)} new testing tasks")
        for t in tasks:
            if t not in result.taskSolutions:
                result.taskSolutions[t] = Frontier([],
                            task=t)
            if t not in result.allFrontiers:
                result.allFrontiers[t] =  Frontier([],task=t)
        for t in tasks + testingTasks:
            if t.name not in result.taskLanguage:
                result.taskLanguage[t.name] = []
    else:  # Start from scratch
        #for graphing of testing tasks
        numTestingTasks = len(testingTasks) if len(testingTasks) != 0 else None
        result = ECResult(parameters=parameters,            
                          grammars=[grammar],
                          taskSolutions={
                              t: Frontier([],
                                          task=t) for t in tasks},
                          models=[],
                          recognitionModel=None, # Backwards compatability
                          numTestingTasks=numTestingTasks,
                          allFrontiers={
                              t: Frontier([],
                                          task=t) for t in tasks},
                          taskLanguage={
                              t.name: [] for t in tasks + testingTasks},
                          tasksAttempted=set())
                          
    # Preload language dataset if avaiable.
    if languageDataset is not None:
        result.languageDatasetPath = languageDatasetDir
        # TODO: figure out how to specify which tasks to load for.
        # May need to separately specify train and test.
        result.taskLanguage, result.vocabularies = languageForTasks(languageDataset, languageDatasetDir, result.taskLanguage)
        if condition_independently_on_language_descriptions:
            tasks, testingTasks = generate_independent_tasks_for_language_descriptions(result, tasks, testingTasks)
        eprint("Loaded language dataset from ", languageDataset)
        if test_task_language: 
            yield result # Integration test outpoint.
    
    if parser == 'loglinear':
        parserModel = LogLinearBigramTransitionParser
    else:
        eprint("Invalid parser: " + parser + ", aborting.")
        assert False
    
    if language_encoder is not None:
        if language_encoder == 'ngram':
            language_encoder = NgramFeaturizer
        elif language_encoder == 'recurrent':
            language_encoder = TokenRecurrentFeatureExtractor
        else:
            eprint("Invalid language encoder: " + language_encoder + ", aborting.")
            assert False

    # Check if we are just updating the full task metrics
    # TODO: this no longer applies (Cathy Wong)
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
        if recognition_0: ECResult.clearRecognitionModel(path)
            
        sys.exit(0)
        
    # Preload any supervision if available into the all frontiers.
    print(f"Found n={len([t for t in tasks if t.add_as_supervised])} supervised tasks; initializing frontiers.")
    for t in tasks:
        if t.add_as_supervised:
            result.allFrontiers[t] = result.allFrontiers[t].combine(Frontier.makeFrontierFromSupervised(t)).topK(maximumFrontier)
    
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
    elif taskReranker == 'curriculum':
        taskBatcher = CurriculumTaskBatcher()
    elif taskReranker == 'sentence_length':
        taskBatcher = SentenceLengthTaskBatcher(tasks, result.taskLanguage)
    else:
        eprint("Invalid task reranker: " + taskReranker + ", aborting.")
        assert False
    
    ######## Test Evaluation and background Helmholtz enumeration.
    for j in range(resume or 0, iterations):
        # Clean up -- at each iteration, remove Helmholtz from the task language dictionary.
        for key_name in list(result.taskLanguage.keys()):
            if "Helmholtz" in key_name:
                result.taskLanguage.pop(key_name)
        print(f"After removing Helmholtz frontiers, task language length is back to {len(result.taskLanguage)} tasks.")
        
        if storeTaskMetrics and rewriteTaskMetrics:
            eprint("Resetting task metrics for next iteration.")
            result.recognitionTaskMetrics = {}

        reportMemory()

        # Evaluate on held out tasks if we have them
        should_skip_test = False
        if testingTimeout > 0 and j == 0 and skip_first_test:
            eprint("SKIPPING FIRST TESTING FOR NOW")
            should_skip_test = True
        elif j == resume and skip_first_test:
            eprint("SKIPPING FIRST TESTING FOR NOW")
            should_skip_test = True
            
        if (not should_skip_test) and testingTimeout > 0 and ((j % testEvery == 0) or (j == iterations - 1)):
            eprint("Evaluating on held out testing tasks for iteration: %d" % (j))
            evaluateOnTestingTasks(result, testingTasks, grammar,
                                   CPUs=CPUs, maximumFrontier=maximumFrontier,
                                   solver=solver,
                                   enumerationTimeout=testingTimeout, evaluationTimeout=evaluationTimeout,
                                   test_dsl_only=test_dsl_only,
                                   max_mem_per_enumeration_thread=max_mem_per_enumeration_thread)            
        # If we have to also enumerate Helmholtz frontiers,
        # do this extra sneaky in the background
        if n_models > 0 and biasOptimal and helmholtzRatio > 0 and \
           all( str(p) != "REAL" for p in grammar.primitives ): # real numbers don't support this
            # the DSL is fixed, so the dreams are also fixed. don't recompute them.
            if useDSL or 'helmholtzFrontiers' not in locals():
                serialize_special = featureExtractor.serialize_special if hasattr(featureExtractor, 'serialize_special') else None
                maximum_helmholtz = featureExtractor.maximum_helmholtz if hasattr(featureExtractor, 'maximum_helmholtz') else None
                helmholtzFrontiers = backgroundHelmholtzEnumeration(tasks, grammar, enumerationTimeout,
                                                                    evaluationTimeout=evaluationTimeout,
                                                                    special=featureExtractor.special,
                                                                    executable='helmholtz',
                                                                    serialize_special=serialize_special,
                                                                    maximum_size=maximum_helmholtz)
                if test_background_helmholtz: # Integration test exitpoint for testing frontiers.
                    yield helmholtzFrontiers
            else:
                print("Reusing dreams from previous iteration.")
        else:
            helmholtzFrontiers = lambda: []

        reportMemory()
        
        wakingTaskBatch = taskBatcher.getTaskBatch(result, tasks, taskBatchSize, j)
        eprint("Using a waking task batch of size: " + str(len(wakingTaskBatch)))

        # WAKING UP
        if useDSL and enumerationTimeout > 0:
            enumeration_time = enumerationTimeout
            if initialTimeout is not None and initialTimeoutIterations is not None:
                if j < initialTimeoutIterations:
                    eprint(f"Found an annealing schedule; using {initialTimeout}s enumeration.")
                    enumeration_time = initialTimeout
            result.tasksAttempted.update(wakingTaskBatch)
            wake_generative = custom_wake_generative if custom_wake_generative is not None else default_wake_generative
            topDownFrontiers, times = wake_generative(grammar, wakingTaskBatch,
                                                      solver=solver,
                                                      maximumFrontier=maximumFrontier,
                                                      enumerationTimeout=enumeration_time,
                                                      CPUs=CPUs,
                                                      evaluationTimeout=evaluationTimeout,
                                                      max_mem_per_enumeration_thread=max_mem_per_enumeration_thread)
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
        if test_wake_generative_enumeration: yield result
        
        #### Recognition model round 0. No language.
        result.models = [] # Reset the list of models at each iteration.
        if len(recognition_0) > 0 and recognitionTimeout > 0:
            result.tasksAttempted.update(wakingTaskBatch)
            recognition_iteration = 0
            # Should we initialize the weights to be what they were before?
            previousRecognitionModel = None
            if reuseRecognition and result.recognitionModel is not None:
                previousRecognitionModel = result.recognitionModel

            thisRatio = helmholtzRatio
            if j == 0 and not biasOptimal: thisRatio = 0
            if all( f.empty for f in result.allFrontiers.values() ): thisRatio = 1.                

            enumeration_time = enumerationTimeout
            if initialTimeout is not None and initialTimeoutIterations is not None:
                if j < initialTimeoutIterations:
                    eprint(f"Found an annealing schedule; using {initialTimeout}s enumeration.")
                    enumeration_time = initialTimeout
                    
            tasks_hit_recognition_0 = \
             sleep_recognition(result, grammar, wakingTaskBatch, tasks, testingTasks, result.allFrontiers.values(),
                               ensembleSize=ensembleSize, 
                               mask=mask,
                               activation=activation, contextual=contextual, biasOptimal=biasOptimal,
                               previousRecognitionModel=previousRecognitionModel, matrixRank=matrixRank,
                               timeout=recognitionTimeout, evaluationTimeout=evaluationTimeout,
                               enumerationTimeout=enumeration_time,
                               helmholtzRatio=thisRatio, helmholtzFrontiers=helmholtzFrontiers(),
                               auxiliaryLoss=auxiliaryLoss, cuda=cuda, CPUs=CPUs, solver=solver,
                               recognitionSteps=recognitionSteps, maximumFrontier=maximumFrontier,
                               featureExtractor=featureExtractor, 
                               language_encoder=language_encoder,
                               recognitionEpochs=recognitionEpochs[recognition_iteration],
                               recognition_iteration=recognition_iteration,
                               recognition_inputs=model_inputs[recognition_iteration],
                               finetune_from_example_encoder=finetune_1,
                               language_data=None,
                               language_lexicon=None,
                               test_only_after_recognition=test_only_after_recognition,
                               pretrained_word_embeddings=pretrained_word_embeddings,
                               max_mem_per_enumeration_thread=max_mem_per_enumeration_thread)

            showHitMatrix(tasksHitTopDown, tasks_hit_recognition_0, wakingTaskBatch)
            
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
        if test_sleep_recognition_0: yield result
        
        ### Induce synchronous grammar for generative model with language.
        # We use this to pre-generate information that can be used to label the Helmholtz samples.
        translation_info = None
        if "language" in recognition_1 and synchronous_grammar:
            if all( f.empty for f in result.allFrontiers.values() ):
                eprint("No non-empty frontiers to train a translation model, skipping.")
            else:
                translation_info = induce_synchronous_grammar(frontiers=result.allFrontiers.values(),
                                tasks=tasks, testingTasks=testingTasks, tasksAttempted=result.tasksAttempted,
                                grammar=grammar, 
                                language_encoder=language_encoder, 
                                language_data=result.taskLanguage,
                                language_lexicon=result.vocabularies["train"],
                                output_prefix=outputPrefix,
                                moses_dir=moses_dir,
                                max_phrase_length=smt_phrase_length,
                                pseudoalignments=smt_pseudoalignments,
                                debug=debug,
                                iteration=j)
        
        #### Recognition model round 1. With language, using the joint generative model to label the Helmholtz samples.
        if len(recognition_1) > 0:
            result.tasksAttempted.update(wakingTaskBatch)
            recognition_iteration = 1
            thisRatio = helmholtzRatio
            if j == 0 and not biasOptimal: thisRatio = 0
            if all( f.empty for f in result.allFrontiers.values() ): thisRatio = 1.                
            
            enumeration_time = enumerationTimeout
            if initialTimeout is not None and initialTimeoutIterations is not None:
                if j < initialTimeoutIterations:
                    eprint(f"Found an annealing schedule; using {initialTimeout}s enumeration.")
                    enumeration_time = initialTimeout
                    
            tasks_hit_recognition_1 = \
             sleep_recognition(result, grammar, wakingTaskBatch, tasks, testingTasks, result.allFrontiers.values(),
                               ensembleSize=ensembleSize, 
                               mask=mask,
                               activation=activation, contextual=contextual, biasOptimal=biasOptimal,
                               previousRecognitionModel=None, matrixRank=matrixRank,
                               timeout=recognitionTimeout, evaluationTimeout=evaluationTimeout,
                               enumerationTimeout=enumeration_time,
                               helmholtzRatio=thisRatio, helmholtzFrontiers=helmholtzFrontiers(),
                               auxiliaryLoss=auxiliaryLoss, cuda=cuda, CPUs=CPUs, solver=solver,
                               recognitionSteps=recognitionSteps, maximumFrontier=maximumFrontier,
                               featureExtractor=featureExtractor, 
                               language_encoder=language_encoder,
                               recognitionEpochs=recognitionEpochs[recognition_iteration],
                               recognition_iteration=recognition_iteration,
                               recognition_inputs=model_inputs[recognition_iteration],
                               finetune_from_example_encoder=finetune_1,
                               language_data=result.taskLanguage,
                               language_lexicon=result.vocabularies["train"],
                               helmholtz_nearest_language=helmholtz_nearest_language,
                               helmholtz_translation_info=translation_info,
                               test_only_after_recognition=test_only_after_recognition,
                               pretrained_word_embeddings=pretrained_word_embeddings,
                               max_mem_per_enumeration_thread=max_mem_per_enumeration_thread)

            showHitMatrix(tasksHitTopDown, tasks_hit_recognition_1, wakingTaskBatch)
            
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
        
        if test_sleep_recognition_1: yield result
        
        # Interactive mode.
        if interactive or useWakeLanguage:
            tasks_hit_parser = default_wake_language(grammar, wakingTaskBatch,
                                    testingTasks=testingTasks,
                                    maximumFrontier=maximumFrontier,
                                    enumerationTimeout=enumerationTimeout,
                                    CPUs=CPUs,
                                    solver=solver,
                                    parser=parserModel,
                                    interactiveTasks=interactiveTasks,
                                    evaluationTimeout=evaluationTimeout,
                                    currentResult=result,
                                    interactive=interactive,
                                    language_encoder=language_encoder,
                                    cuda=cuda,
                                    epochs=recognitionEpochs)
                                    
            for task in tasks_hit_parser:
                if task not in result.allFrontiers: continue
                result.allFrontiers[task] = result.allFrontiers[task].\
                                            combine(grammar.rescoreFrontier(tasks_hit_parser[task])).\
                                            topK(maximumFrontier)
            
            eprint("Frontiers discovered with parser: " + str(len(tasks_hit_parser)))
            eprint("Total frontiers: " + str(len([f for f in result.allFrontiers.values() if not f.empty])))
            eprint(Frontier.describe([f for f in tasks_hit_parser.values() if not f.empty]))
            showHitMatrix(tasksHitTopDown, set(tasks_hit_parser.keys()), wakingTaskBatch)
            
        # Sleep-G
        if useDSL and not(noConsolidation) or debug:
            eprint(f"Currently using this much memory: {getThisMemoryUsage()}")
            ## Language for compression
            language_alignments = []
            if language_compression:
                eprint(f"Using language alignments for compression.")
                if debug: 
                    eprint(f"Running in debug -- using an old checkpoint for alignments.")
                    output_dir = "experimentOutputs/clevr/2020-05-21T17-56-08-470454/moses_corpus_1"
                else: 
                    eprint(f"Reading alignments from the Moses dir.")
                    if translation_info is None:
                        eprint(f"No translation info found; setting output dir to None.")
                        output_dir = None
                    else:
                        output_dir = translation_info["output_dir"]
                language_alignments = get_alignments(grammar=grammar, output_dir=output_dir)
                
            grammar = consolidate(result, grammar, topK=topK, pseudoCounts=pseudoCounts, arity=arity, aic=aic,
                                  structurePenalty=structurePenalty, compressor=compressor, CPUs=CPUs,
                                  iteration=j, language_alignments=language_alignments,
                                  lc_score=lc_score,
                                  max_compression=max_compression)
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
            if recognition_0:
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
                           CPUs=None, solver=None, maximumFrontier=None, enumerationTimeout=None, evaluationTimeout=None,
                           test_dsl_only= False,max_mem_per_enumeration_thread=1000000):
    
    if len(result.models) > 0 and not test_dsl_only:
        eprint("Evaluating on testing tasks using the recognizer.")
        recognizer = result.models[-1]
        testingFrontiers, times = \
         recognizer.enumerateFrontiers(testingTasks, 
                                       CPUs=CPUs,
                                       solver=solver,
                                       maximumFrontier=maximumFrontier,
                                       enumerationTimeout=enumerationTimeout,
                                       evaluationTimeout=evaluationTimeout,
                                       testing=True,
                                       max_mem_per_enumeration_thread=max_mem_per_enumeration_thread)
        updateTaskSummaryMetrics(result.recognitionTaskMetrics, recognizer.taskGrammarLogProductions(testingTasks), 'heldoutTaskLogProductions')
        updateTaskSummaryMetrics(result.recognitionTaskMetrics, recognizer.taskGrammarEntropies(testingTasks), 'heldoutTaskGrammarEntropies')
        updateTaskSummaryMetrics(result.recognitionTaskMetrics, recognizer.taskGrammarEntropies(testingTasks), 'heldoutTaskGrammarEntropies')
    else:
        if test_dsl_only:
            eprint("Evaluating on testing tasks using the following DSL:")
        testingFrontiers, times = multicoreEnumeration(grammar, testingTasks, 
                                                       solver=solver,
                                                       maximumFrontier=maximumFrontier,
                                                       enumerationTimeout=enumerationTimeout,
                                                       CPUs=CPUs,
                                                       evaluationTimeout=evaluationTimeout,
                                                       testing=True,
                                                       max_mem_per_enumeration_thread=max_mem_per_enumeration_thread)
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

def default_wake_language(grammar, tasks, 
                    testingTasks,
                    maximumFrontier=None,
                    enumerationTimeout=None,
                    CPUs=None,
                    solver=None,
                    parser=None,
                    interactiveTasks=None,
                    evaluationTimeout=None,
                    currentResult=None,
                    get_language_fn=None,
                    interactive=None,
                    language_encoder=None,
                    program_featurizer=None,
                    epochs=None,
                    cuda=False,
                    max_mem_per_enumeration_thread=1000000):
    # Get interactive descriptions for all solutions.
    if get_language_fn is not None:
        solutions = [f for f in currentResult.allFrontiers.values() if not f.empty]
        solutions_language = get_language_fn(solutions)
        for t in solutions_language:
            currentResult.taskLanguage[t].append(solutions_language[t])
    # Retrain the parser.
    parser_model = parser(grammar, 
                          language_data=currentResult.taskLanguage,
                          frontiers=currentResult.allFrontiers,
                          tasks=tasks,
                          testingTasks=testingTasks,
                          cuda=cuda,
                          language_feature_extractor=language_encoder)
    parser_model.train(epochs=epochs)
    
    if interactive:
        eprint("Not yet implemented: interactive mode.")
        assert False
    else:
        # Enumerative search using the retrained parser.
        eprint("Enumerating from the trained parser.")
        enumerated_frontiers, recognition_times = parser_model.enumerateFrontiers(tasks=tasks, 
                                        enumerationTimeout=enumerationTimeout,
                                        solver=solver,
                                        CPUs=CPUs,
                                        maximumFrontier=maximumFrontier,
                                        evaluationTimeout=evaluationTimeout)
        tasks_hit = {f.task : f for f in enumerated_frontiers if not f.empty}
        currentResult.parser = parser_model
        return tasks_hit
    
        
def default_wake_generative(grammar, tasks, 
                    maximumFrontier=None,
                    enumerationTimeout=None,
                    CPUs=None,
                    solver=None,
                    evaluationTimeout=None,
                    max_mem_per_enumeration_thread=1000000):
    topDownFrontiers, times = multicoreEnumeration(grammar, tasks, 
                                                   maximumFrontier=maximumFrontier,
                                                   enumerationTimeout=enumerationTimeout,
                                                   CPUs=CPUs,
                                                   solver=solver,
                                                   evaluationTimeout=evaluationTimeout,
                                                   max_mem_per_enumeration_thread=max_mem_per_enumeration_thread)
    eprint("Generative model enumeration results:")
    eprint(Frontier.describe(topDownFrontiers))
    summaryStatistics("Generative model", [t for t in times.values() if t is not None])
    return topDownFrontiers, times

def induce_synchronous_grammar(frontiers, tasks, testingTasks, tasksAttempted, grammar, 
                    language_encoder=None, language_data=None, language_lexicon=None,
                    output_prefix=None,
                    moses_dir=None,
                    max_phrase_length=None,
                    pseudoalignments=None,
                    debug=None,
                    iteration=None):    
    encoder = language_encoder(tasks, testingTasks=testingTasks, cuda=False, language_data=language_data, lexicon=language_lexicon)
    n_frontiers = len([f for f in frontiers if not f.empty])
    eprint(f"Inducing synchronous grammar. Using n=[{n_frontiers} frontiers],")
    if n_frontiers == 0:    
        return None
    if debug:
        # Use the previous iteration so that we can peek into it
        debug_iteration = iteration - 1
        corpus_dir = os.path.join(os.path.dirname(output_prefix), f'moses_corpus_{debug_iteration}')
        # eprint(f"Running in non-debug mode, writing corpus files to {corpus_dir}.")
        # eprint("Running in debug mode, writing corpus files to tmp.")
        # corpus_dir = os.path.split(os.path.dirname(output_prefix))[0] # Remove timestamp and type prefix on checkpoint
        # corpus_dir = os.path.join(corpus_dir, 'corpus_tmp')
    else:
        corpus_dir = os.path.join(os.path.dirname(output_prefix), f'moses_corpus_{iteration}')
        eprint(f"Running in non-debug mode, writing corpus files to {corpus_dir}.")
    alignment_outputs = smt_alignment(tasks, tasksAttempted, frontiers, grammar, encoder, corpus_dir, moses_dir, phrase_length=max_phrase_length, n_pseudo=pseudoalignments)
    return alignment_outputs

def sleep_recognition(result, grammar, taskBatch, tasks, testingTasks, allFrontiers, _=None,
                      ensembleSize=1, featureExtractor=None, matrixRank=None, mask=False,
                      activation=None, contextual=True, biasOptimal=True,
                      previousRecognitionModel=None, recognitionSteps=None,
                      timeout=None, enumerationTimeout=None, evaluationTimeout=None,
                      helmholtzRatio=None, helmholtzFrontiers=None, maximumFrontier=None,
                      auxiliaryLoss=None, cuda=None, CPUs=None, solver=None,
                      language_encoder=None,
                      recognitionEpochs=None,
                      recognition_iteration=None,
                      recognition_inputs=[],
                      finetune_from_example_encoder=False,
                      language_data=None,
                      language_lexicon=None,
                      helmholtz_nearest_language=0,
                      helmholtz_translation_info=None,
                      test_only_after_recognition=False,
                      pretrained_word_embeddings=None,
                      max_mem_per_enumeration_thread=1000000):
    ### Pre-check: have we discovered any program solutions on the training set?
    ## If not, we have no data from which to train a joint language-example-based model, so we skip this round if you required training on both language and examples.
    n_frontiers = len([f for f in allFrontiers if not f.empty])
    eprint(f"Recognition iteration [{recognition_iteration}]. Attempting training using: {[recognition_inputs]} on {n_frontiers}.")
    if not 'examples' in recognition_inputs and 'language' in recognition_inputs and n_frontiers < 1:
        result.models += [None]
        eprint(f"! No non-empty language frontiers to train on. Skipping language-only enumeration.")
        result.hitsAtEachWake.append(0)
        return set()
    
    example_encoders, language_encoders = [None] * ensembleSize, [None] * ensembleSize
    pretrained_model = None

    if 'examples' in recognition_inputs:
        # Initialize the I/O example encoders. We pass in all of the tasks in the entire training set at once, which are used to pre-calculate Helmholtz inputs, language, and other dataset-based statistics. 
        example_encoders = [featureExtractor(tasks, testingTasks=testingTasks, cuda=cuda) for i in range(ensembleSize)]
        if recognition_iteration > 0 and finetune_from_example_encoder:
            eprint("Finetuning from the previous iteration's example encoder and model.")
            pretrained_model = result.models[recognition_iteration - 1]
            assert(pretrained_model.featureExtractor is not None)
    if 'language' in recognition_inputs:
        # Initialize the language-only example encoders.
        language_encoders = [language_encoder(tasks, testingTasks=testingTasks, cuda=cuda, language_data=language_data, lexicon=language_lexicon, smt_translation_info=helmholtz_translation_info,
        pretrained_word_embeddings=pretrained_word_embeddings) for i in range(ensembleSize)]
    if recognition_iteration > 0 and helmholtz_nearest_language > 0:
        # This labels helmholtz with the 'nearest' language. It is an experimental feature that we do not use in the released papers.
        nearest_encoder = result.models[recognition_iteration - 1]
        nearest_tasks = tasks
        helmholtzFrontiers = [] # For now, reset the frontiers so we're guaranteed to freshly name them
    else:
        nearest_encoder, nearest_tasks = None, None
        helmholtz_nearest_language = 0
    
    # Initializes the full recognition model architecture.
    recognizers = [RecognitionModel(example_encoder=example_encoders[i],
                                    language_encoder=language_encoders[i],
                                    grammar=grammar,
                                    mask=mask,
                                    rank=matrixRank,
                                    activation=activation,
                                    cuda=cuda,
                                    contextual=contextual,
                                    pretrained_model=pretrained_model,
                                    helmholtz_nearest_language=helmholtz_nearest_language,
                                    helmholtz_translations=helmholtz_translation_info, # This object contains information for using the joint generative model over programs and language.
                                    nearest_encoder=nearest_encoder,
                                    nearest_tasks=nearest_tasks,
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
                                                                         vectorized=True,
                                                                         epochs=recognitionEpochs),
                                     recognizers,
                                     seedRandom=True)
    eprint(f"Currently using this much memory: {getThisMemoryUsage()}")
    
    if test_only_after_recognition:
        eprint("Trained an ensemble of %d recognition models, now testing the first one.")
        result.models += [trainedRecognizers[0]]
        evaluateOnTestingTasks(result, testingTasks, grammar,
                               CPUs=CPUs, maximumFrontier=maximumFrontier,
                               solver=solver,
                               enumerationTimeout=enumerationTimeout, evaluationTimeout=evaluationTimeout,
                               test_dsl_only=False,
                               max_mem_per_enumeration_thread=max_mem_per_enumeration_thread)   
        
        sys.exit(0)
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
                                                      solver=solver,
                                                      max_mem_per_enumeration_thread=max_mem_per_enumeration_thread)
        ensembleFrontiers.append(bottomupFrontiers)
        ensembleTimes.append([t for t in allRecognitionTimes.values() if t is not None])
        ensembleRecognitionTimes.append(allRecognitionTimes)

        recognizerTasksHitBottomUp = {f.task for f in bottomupFrontiers if not f.empty}
        totalTasksHitBottomUp.update(recognizerTasksHitBottomUp)
        eprint("Recognizer %d solved %d/%d tasks; total tasks solved is now %d." % (recIndex, len(recognizerTasksHitBottomUp), len(tasks), len(totalTasksHitBottomUp)))
        if len(recognizerTasksHitBottomUp) >= mostTasks:
            # TODO (cathywong): could consider keeping the one that put the highest likelihood on the solved tasks.
            bestRecognizer = recIndex

    
    result.models += [trainedRecognizers[bestRecognizer]]
    eprint(f"Recognition models contains: n={len(result.models)} models")
    
    # Store the recognizer that discovers the most frontiers in the result.
    eprint("Best recognizer: %d." % bestRecognizer)
    # result.recognitionModel = trainedRecognizers[bestRecognizer]
    # result.trainSearchTime = {tk: tm for tk, tm in ensembleRecognitionTimes[bestRecognizer].items()
    #                           if tm is not None}
    # updateTaskSummaryMetrics(result.recognitionTaskMetrics, ensembleRecognitionTimes[bestRecognizer], 'recognitionBestTimes')
    # updateTaskSummaryMetrics(result.recognitionTaskMetrics, result.recognitionModel.taskHiddenStates(tasks), 'hiddenState')
    # updateTaskSummaryMetrics(result.recognitionTaskMetrics, result.recognitionModel.taskGrammarLogProductions(tasks), 'taskLogProductions')
    # updateTaskSummaryMetrics(result.recognitionTaskMetrics, result.recognitionModel.taskGrammarEntropies(tasks), 'taskGrammarEntropies')
    # if contextual:
    #     updateTaskSummaryMetrics(result.recognitionTaskMetrics,
    #                              result.recognitionModel.taskGrammarStartProductions(tasks),
    #                              'startProductions')

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
                structurePenalty=None, compressor=None, CPUs=None, iteration=None, language_alignments=None, lc_score=0.0,
                max_compression=1000):
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
                                                      backend=compressor, CPUs=CPUs, iteration=iteration,
                                                      language_alignments=language_alignments,
                                                      executable="compression",
                                                      lc_score=lc_score,max_compression=max_compression)
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
                         recognition_0=["examples"],
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
    ## Recognition models.
    parser.add_argument("--recognition_0",
                        dest="recognition_0", 
                        default=["examples"],
                        nargs="*",
                        help="""0th recognition model. Specify a list of features to use. Choices: ["examples"]
                        Default: %s""" % recognition_0)
    parser.add_argument("--recognition_1",
                        dest="recognition_1", 
                        default=[],
                        nargs="*",
                        help="""1st recognition model. Specify a list of features to use. Choices: ["examples", "language"]
                        Default: %s""" % recognition_0)
    parser.add_argument("--finetune_1",
                        dest="finetune_1",
                        action="store_true",
                        help="""If true, finetunes recognition 1 from recognition 0.""")
    parser.add_argument("-RE", "--recognitionEpochs",
                        default=[None],
                        nargs="*",
                        help="List of number of epochs to train the recognition model at each round. Can be specified instead of train time or gradient steps.",
                        type=int)
    parser.add_argument("-r",
                        "--Helmholtz",
                        dest="helmholtzRatio",
                        help="""When training recognition models, what fraction of the training data should be samples from the generative model? Default %f""" %
                        helmholtzRatio,
                        default=helmholtzRatio,
                        type=float)
    parser.add_argument("--biasOptimal",
                        help="Enumerate dreams rather than sample them & bias-optimal recognition objective",
                        default=False, action="store_true")
    parser.add_argument("--contextual",
                        help="bigram recognition model (default is unigram model)",
                        default=False, action="store_true")
    parser.add_argument("--ensembleSize",
                        dest="ensembleSize",
                        default=1,
                        help="Number of recognition models to train and enumerate from at each iteration.",
                        type=int)
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
    
    ### SMT generative model training.
    parser.add_argument("--lc_score",
                        default=0.2,
                        type=float,
                        help="Amount to factor language into language score.")
    parser.add_argument("--max_compression",
                        default=5,
                        type=float,
                        help="Maximum number of iterations to run the compression loop for at each iteration.")
    parser.add_argument("--language_compression",
                        action='store_true',
                        help="Whether to use the synchronous grammar during compression.")
    parser.add_argument("--synchronous_grammar",
                        action='store_true',
                        help="Whether to train a synchronous grammar over the programs and language..")
    parser.add_argument("--moses_dir",
        default="../moses_compiled",
        help="Location of top-level Moses SMT directory for machine translation.")
    parser.add_argument("--smt_phrase_length",
        default=5,
        type=int,
        help="Maximum phrase length when learning Moses phrase model.")
    parser.add_argument("--smt_pseudoalignments",
        default=0,
        type=float,
        help="Pseudoalignments count for generative model.")
    
    parser.add_argument("--pretrained_word_embeddings",
                        action="store_true",
                        help="Whether to use pretrained word embeddings to initialize the parser.")

    parser.add_argument("--helmholtz_nearest_language",
                        dest="helmholtz_nearest_language",
                        default=0, 
                        type=int)
    parser.add_argument("--language_encoder",
                        dest="language_encoder",
                        help="Which language encoder to use.",
                        choices=["ngram",
                                "recurrent"],
                        default=None,
                        type=str)
    parser.add_argument("--languageDataset",
                        dest="languageDataset",
                        help="Name of language dataset or datasets if using language features.",
                        nargs="+",
                        default=None,
                        type=str)
    parser.add_argument("--condition_independently_on_language_descriptions",
                        action='store_true',
                        help="If true, treats each linguistic description provided as a separate task and searches separately on it.")
                        
    ### Algorithm training details.
    parser.add_argument("--max_mem_per_enumeration_thread",	
                        default=1000000000,	
                        type=int,
                        help="""The maximum memory to allow for an enumeration thread.""")
    parser.add_argument("--skip_first_test",	
                        action="store_true",	
                        dest="skip_first_test",	
                        help="""Skip the first testing round to avoid redundancy.""")
    parser.add_argument("--test_only_after_recognition",	
                        action="store_true",	
                        dest="test_only_after_recognition",	
                        help="""Trains the recognition model and then runs tests instead of searching. Used for training different recognition models.""")
    parser.add_argument("--test_dsl_only",	
                        action="store_true",	
                        dest="test_dsl_only",	
                        help="""Force uses the DSL for enumerative testing.""")
                        
    parser.add_argument("--debug",
                        action="store_true",
                        dest="debug",
                        help="""General purpose debug flag.""")
    parser.add_argument("--resume",
                        help="Resumes EC algorithm from checkpoint. You can either pass in the path of a checkpoint, or you can pass in the iteration to resume from, in which case it will try to figure out the path.",
                        default=None,
                        type=str)
    parser.add_argument("-i", "--iterations",
                        help="default: %d" % iterations,
                        default=iterations,
                        type=int)
    parser.add_argument("--initialTimeout",
                        default=None,
                        help="In seconds.",
                        type=int)
    parser.add_argument("--initialTimeoutIterations",
                        default=None,
                        help="How many iterations to use an initial timeout for.",
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
                        help="Number of gradient steps to train the recognition model. Can be specified instead of train time or epochs.",
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
        "--compressor",
        default=compressor,
        choices=["pypy","rust","vs","pypy_vs","ocaml", "python", "stitch"])
    parser.add_argument(
        "--matrixRank",
        help="Maximum rank of bigram transition matrix for contextual recognition model. Defaults to full rank.",
        default=None,
        type=int)
    parser.add_argument(
        "--mask",
        help="Unconditional bigram masking",
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
            "randomLowEntropykNN",
            "curriculum",
            "sentence_length"],
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
    parser.add_argument("--interactive",
                        action="store_true", default=False,
                        dest='interactive',
                        help="Add an interactive wake generative cycle.")  
    parser.add_argument("--interactiveTasks",
                        dest="interactiveTasks",
                        help="Reranking function used to order the tasks we train on during waking.",
                        choices=["random"],
                        default='random',
                        type=str)
    parser.add_argument("--parser",
                        dest="parser",
                        help="Semantic parser for interactive mode.",
                        choices=["loglinear"],
                        default='loglinear',
                        type=str)              
    parser.set_defaults(recognition_0=recognition_0,
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

    if v["recognition_0"] and v["recognitionTimeout"] is None:
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

