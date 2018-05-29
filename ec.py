
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
                 baselines=None):
        self.testingSearchTime = testingSearchTime or []
        self.searchTimes = searchTimes or []
        self.recognitionModel = recognitionModel
        self.averageDescriptionLength = averageDescriptionLength or []
        self.parameters = parameters
        self.learningCurve = learningCurve or []
        self.grammars = grammars or []
        self.taskSolutions = taskSolutions or {}
        # baselines is a dictionary of name -> ECResult
        self.baselines = baselines or {}

    def __repr__(self):
        attrs = ["{}={}".format(k, v) for k, v in self.__dict__.items()]
        return "ECResult({})".format(", ".join(attrs))

    # Linux does not like files that have more than 256 characters
    # So when exporting the results we abbreviate the parameters
    abbreviations = {"frontierSize": "fs",
                     "iterations": "it",
                     "maximumFrontier": "MF",
                     "onlyBaselines": "baseline",
                     "pseudoCounts": "pc",
                     "structurePenalty": "L",
                     "helmholtzRatio": "HR",
                     "topK": "K",
                     "enumerationTimeout": "ET",
                     "useRecognitionModel": "rec"}

    @staticmethod
    def abbreviate(parameter): return ECResult.abbreviations.get(
        parameter, parameter)

    @staticmethod
    def parameterOfAbbreviation(abbreviation):
        return ECResult.abbreviationToParameter.get(abbreviation, abbreviation)


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
               testingTasks=[],
               benchmark=None,
               iterations=None,
               resume=None,
               frontierSize=None,
               enumerationTimeout=None,
               testingTimeout=None,
               expandFrontier=None,
               resumeFrontierSize=None,
               useRecognitionModel=True,
               useNewRecognitionModel=False,
               steps=250,  # 250,
               helmholtzRatio=0.,
               helmholtzBatch=5000,
               featureExtractor=None,
               activation='relu',
               topK=1,
               topk_use_map=True,
               maximumFrontier=None,
               pseudoCounts=1.0, aic=1.0,
               structurePenalty=0.001, arity=0,
               evaluationTimeout=1.0,  # seconds
               CPUs=1,
               cuda=False,
               message="",
               onlyBaselines=False,
               outputPrefix=None):
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

    if testingTimeout > 0 and len(testingTasks) == 0:
        eprint("You specified a testingTimeout, but did not provide any held out testing tasks, aborting.")
        assert False

    # We save the parameters that were passed into EC
    # This is for the purpose of exporting the results of the experiment
    parameters = {
        k: v for k,
        v in locals().items() if k not in {
            "tasks",
            "grammar",
            "cuda",
            "_",
            "solver",
            "testingTimeout",
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
        for k in {"activation", "helmholtzRatio", "steps"}:
            del parameters[k]

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
            path = checkpointPath(0, extra="_baselines")
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
    eprint()

    # Restore checkpoint
    if resume is not None:
        path = checkpointPath(
            resume, extra="_baselines" if onlyBaselines else "")
        with open(path, "rb") as handle:
            result = dill.load(handle)
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
                                            if not str(p).startswith("fix")}))
            if compressor == "rust":
                eprint(
                    "Rust compressor is currently not compatible with bootstrapping.",
                    "Falling back on pypy compressor.")
                compressor = "pypy"
        result = ECResult(
            parameters=parameters,
            grammars=[grammar],
            taskSolutions={
                t: Frontier(
                    [],
                    task=t) for t in tasks},
            recognitionModel=None)

    # just plopped this in here, hope it works: -it doesn't. having issues.
    if useNewRecognitionModel and (
        not hasattr(
            result,
            'recognitionModel') or not isinstance(
            result.recognitionModel,
            NewRecognitionModel)):
        eprint("Creating new recognition model")
        featureExtractorObject = featureExtractor(tasks + testingTasks)
        result.recognitionModel = NewRecognitionModel(
            featureExtractorObject, grammar, cuda=cuda)
    # end

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

    if solver == 'python':
        useMultithread = True
        eprint("WARNING: switching to older 'multithreaded' enumeration code for compatability with python solver")
    else:
        useMultithread = False

    for j in range(resume or 0, iterations):

        # Evaluate on held out tasks if we have them
        if testingTimeout > 0:
            eprint("Evaluating on held out testing tasks.")
            if useRecognitionModel and j > 0:
                _, times = result.recognitionModel.enumerateFrontiers(testingTasks, likelihoodModel,
                                                                      CPUs=CPUs,
                                                                      solver=solver,
                                                                      maximumFrontier=maximumFrontier,
                                                                      enumerationTimeout=testingTimeout,
                                                                      evaluationTimeout=evaluationTimeout,
                                                                      testing=True, useMultithread=useMultithread)
            else:
                if useMultithread:
                    _, times = multithreadedEnumeration(grammar, testingTasks, likelihoodModel,
                                                    solver=solver,
                                                    maximumFrontier=maximumFrontier,
                                                    enumerationTimeout=testingTimeout,
                                                    CPUs=CPUs,
                                                    evaluationTimeout=evaluationTimeout,
                                                    testing=True)                    
                else:
                    _, times = multicoreEnumeration(grammar, testingTasks, likelihoodModel,
                                                    solver=solver,
                                                    maximumFrontier=maximumFrontier,
                                                    enumerationTimeout=testingTimeout,
                                                    CPUs=CPUs,
                                                    evaluationTimeout=evaluationTimeout,
                                                    testing=True)

            eprint(
                "Hits %d/%d testing tasks" %
                (len(times), len(testingTasks)))
            summaryStatistics("Testing tasks", times)
            result.testingSearchTime.append(times)

        if j >= 2 and expandFrontier and result.learningCurve[-1] <= result.learningCurve[-2] and \
           result.learningCurve[-1] < len(tasks):
            oldEnumerationTimeout = enumerationTimeout
            if expandFrontier <= 10:
                enumerationTimeout = int(enumerationTimeout * expandFrontier)
            else:
                enumerationTimeout = int(enumerationTimeout + expandFrontier)
            eprint(
                "Expanding enumeration timeout from {} to {} because of no progress".format(
                    oldEnumerationTimeout,
                    enumerationTimeout))
        if useMultithread:
            frontiers, times = multithreadedEnumeration(grammar, tasks, likelihoodModel,
                                            solver=solver,
                                            frontierSize=frontierSize,
                                            maximumFrontier=maximumFrontier,
                                            enumerationTimeout=enumerationTimeout,
                                            CPUs=CPUs,
                                            evaluationTimeout=evaluationTimeout)
        else:             
            frontiers, times = multicoreEnumeration(grammar, tasks, likelihoodModel,
                                                    solver=solver,
                                                    maximumFrontier=maximumFrontier,
                                                    enumerationTimeout=enumerationTimeout,
                                                    CPUs=CPUs,
                                                    evaluationTimeout=evaluationTimeout)

        if expandFrontier and (not useRecognitionModel) \
           and (j == 0 and times == [] or
                j > 0 and sum(not f.empty for f in frontiers) <= result.learningCurve[-1] and
                result.learningCurve[-1] < len(tasks)):
            timeout = enumerationTimeout
            unsolvedTasks = [f.task for f in frontiers if f.empty]
            while True:
                eprint(
                    "Expanding enumeration timeout from %i to %i because of no progress. Focusing exclusively on %d unsolved tasks." %
                    (timeout, timeout * expandFrontier, len(unsolvedTasks)))
                timeout = timeout * expandFrontier
                if useMultithread:
                    unsolvedFrontiers, unsolvedTimes = \
                        multithreadedEnumeration(grammar, unsolvedTasks, likelihoodModel,
                                                 solver=solver,
                                                 frontierSize=frontierSize,
                                                 maximumFrontier=maximumFrontier,
                                                 enumerationTimeout=timeout,
                                                 CPUs=CPUs,
                                                 evaluationTimeout=evaluationTimeout)                    
                else:
                    unsolvedFrontiers, unsolvedTimes = \
                        multicoreEnumeration(grammar, unsolvedTasks, likelihoodModel,
                                             solver=solver,
                                             maximumFrontier=maximumFrontier,
                                             enumerationTimeout=timeout,
                                             CPUs=CPUs,
                                             evaluationTimeout=evaluationTimeout)
                if any(not f.empty for f in unsolvedFrontiers):
                    times += unsolvedTimes
                    unsolvedFrontiers = {f.task: f for f in unsolvedFrontiers}
                    frontiers = [f if not f.empty else unsolvedFrontiers[f.task]
                                 for f in frontiers]
                    break

        eprint("Generative model enumeration results:")
        eprint(Frontier.describe(frontiers))
        summaryStatistics("Generative model", times)

        tasksHitTopDown = {f.task for f in frontiers if not f.empty}

        # Train + use recognition model
        if useRecognitionModel:
            featureExtractorObject = featureExtractor(tasks + testingTasks)
            recognizer = RecognitionModel(
                featureExtractorObject,
                grammar,
                activation=activation,
                cuda=cuda)

            recognizer.train(frontiers, topK=topK, steps=steps,
                             CPUs=CPUs,
                             timeout=enumerationTimeout,  # give training as much time as enumeration
                             helmholtzBatch=helmholtzBatch,
                             helmholtzRatio=helmholtzRatio if j > 0 or helmholtzRatio == 1. else 0.)
            result.recognitionModel = recognizer

            bottomupFrontiers, times = recognizer.enumerateFrontiers(tasks, likelihoodModel,
                                                                     CPUs=CPUs,
                                                                     solver=solver,
                                                                     frontierSize=frontierSize,
                                                                     maximumFrontier=maximumFrontier,
                                                                     enumerationTimeout=enumerationTimeout,
                                                                     evaluationTimeout=evaluationTimeout,
                                                                     useMultithread=useMultithread)

        elif useNewRecognitionModel:  # Train a recognition model
            result.recognitionModel.updateGrammar(grammar)
            # changed from result.frontiers to frontiers and added thingy
            result.recognitionModel.train(
                frontiers,
                topK=topK,
                steps=steps,
                helmholtzRatio=helmholtzRatio)
            eprint("done training recognition model")
            bottomupFrontiers = result.recognitionModel.enumerateFrontiers(
                tasks,
                likelihoodModel,
                CPUs=CPUs,
                solver=solver,
                maximumFrontier=maximumFrontier,
                frontierSize=frontierSize,
                enumerationTimeout=enumerationTimeout,
                evaluationTimeout=evaluationTimeout,
                useMultithread=useMultithread)
        if useRecognitionModel or useNewRecognitionModel:

            eprint("Recognition model enumeration results:")
            eprint(Frontier.describe(bottomupFrontiers))
            summaryStatistics("Recognition model", times)

            result.averageDescriptionLength.append(mean(-f.marginalLikelihood()
                                                        for f in bottomupFrontiers
                                                        if not f.empty))

            tasksHitBottomUp = {
                f.task for f in bottomupFrontiers if not f.empty}
            showHitMatrix(tasksHitTopDown, tasksHitBottomUp, tasks)
            # Rescore the frontiers according to the generative model
            # and then combine w/ original frontiers
            bottomupFrontiers = [
                grammar.rescoreFrontier(f) for f in bottomupFrontiers]
            frontiers = [
                f.combine(b) for f, b in zip(
                    frontiers, bottomupFrontiers)]
        else:
            result.averageDescriptionLength.append(mean(-f.marginalLikelihood()
                                                        for f in frontiers
                                                        if not f.empty))

        if not useNewRecognitionModel:  # This line is changed, beware
            result.searchTimes.append(times)

            eprint("Average search time: ", int(mean(times) + 0.5),
                   "sec.\tmedian:", int(median(times) + 0.5),
                   "\tmax:", int(max(times) + 0.5),
                   "\tstandard deviation", int(standardDeviation(times) + 0.5))

        # Incorporate frontiers from anything that was not hit
        frontiers = [
            f if not f.empty else grammar.rescoreFrontier(
                result.taskSolutions.get(
                    f.task, Frontier.makeEmpty(
                        f.task))) for f in frontiers]
        frontiers = [f.topK(maximumFrontier) for f in frontiers]

        if maximumFrontier <= 10:
            eprint(
                "Because maximumFrontier is small (<=10), I am going to show you the full contents of all the frontiers:")
            for f in frontiers:
                if f.empty:
                    continue
                eprint(f.task)
                for e in f.normalize():
                    eprint("%.02f\t%s" % (e.logPosterior, e.program))
                eprint()
        # Record the new solutions
        result.taskSolutions = {f.task: f.topK(topK)
                                for f in frontiers}
        result.learningCurve += [
            sum(f is not None and not f.empty for f in result.taskSolutions.values())]

        # Sleep-G
        grammar, frontiers = induceGrammar(grammar, frontiers,
                                           topK=topK, topk_use_map=topk_use_map,
                                           pseudoCounts=pseudoCounts, a=arity,
                                           aic=aic, structurePenalty=structurePenalty,
                                           backend=compressor, CPUs=CPUs, iteration=j)
        result.grammars.append(grammar)
        eprint("Grammar after iteration %d:" % (j + 1))
        eprint(grammar)
        eprint(
            "Expected uses of each grammar production after iteration %d:" %
            (j + 1))
        productionUses = FragmentGrammar.fromGrammar(grammar).\
            expectedUses([f for f in frontiers if not f.empty]).actualUses
        productionUses = {
            p: productionUses.get(
                p, 0.) for p in grammar.primitives}
        for p in sorted(
                productionUses.keys(),
                key=lambda p: -
                productionUses[p]):
            eprint("<uses>=%.2f\t%s" % (productionUses[p], p))
        eprint()
        if maximumFrontier <= 10:
            eprint("Because maximumFrontier is small (<=10), I am going to show you the full contents of all the rewritten frontiers:")
            for f in frontiers:
                if f.empty:
                    continue
                eprint(f.task)
                for e in f.normalize():
                    eprint("%.02f\t%s" % (e.logPosterior, e.program))
                eprint()

        if outputPrefix is not None:
            path = checkpointPath(j + 1)
            with open(path, "wb") as handle:
                try:
                    dill.dump(result, handle)
                except TypeError as e:
                    eprint(result)
                    assert(False)
            eprint("Exported checkpoint to", path)

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
                         topK=1,
                         CPUs=1,
                         useRecognitionModel=True,
                         useNewRecognitionModel=False,
                         steps=250,
                         activation='relu',
                         helmholtzRatio=1.,
                         helmholtzBatch=5000,
                         featureExtractor=None,
                         cuda=None,
                         maximumFrontier=None,
                         pseudoCounts=1.0, aic=1.0,
                         structurePenalty=0.001, a=0,
                         onlyBaselines=False,
                         extras=None):
    if cuda is None:
        cuda = torch.cuda.is_available()
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--resume",
                        help="Resumes EC algorithm from checkpoint",
                        default=None,
                        type=int)
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
    parser.add_argument("--steps", type=int,
                        default=steps,
                        help="""Trainings steps for neural recognition model.
                        Default: %s""" % steps)
    parser.add_argument(
        "--testingTimeout",
        type=int,
        dest="testingTimeout",
        default=0,
        help="Number of seconds we should spend evaluating on each held out testing task.")
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
    parser.set_defaults(useRecognitionModel=useRecognitionModel,
                        featureExtractor=featureExtractor,
                        maximumFrontier=maximumFrontier,
                        cuda=cuda)
    if extras is not None:
        extras(parser)
    v = vars(parser.parse_args())
    return v
