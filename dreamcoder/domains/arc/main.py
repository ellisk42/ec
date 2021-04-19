from collections import defaultdict
import datetime
import dill
import json
import math
import numpy as np
import os
import pickle
import random
import signal
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamcoder.dreaming import helmholtzEnumeration
from dreamcoder.dreamcoder import explorationCompression, sleep_recognition
from dreamcoder.utilities import eprint, flatten, testTrainSplit, lse, runWithTimeout
from dreamcoder.grammar import Grammar, ContextualGrammar
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
from dreamcoder.recognition import RecognitionModel, DummyFeatureExtractor
from dreamcoder.program import Program
from dreamcoder.domains.arc.utilsPostProcessing import *
from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.taskGeneration import *

class EvaluationTimeout(Exception):
    pass


class ArcTask(Task):
    def __init__(self, name, request, examples, evalExamples, features=None, cache=False):
        super().__init__(name, request, examples, features=features, cache=cache)
        self.evalExamples = evalExamples


    def checkEvalExamples(self, e, timeout=None):
        if timeout is not None:
            def timeoutCallBack(_1, _2): raise EvaluationTimeout()
        try:
            signal.signal(signal.SIGVTALRM, timeoutCallBack)
            signal.setitimer(signal.ITIMER_VIRTUAL, timeout)

            try:
                f = e.evaluate([])
            except IndexError:
                # free variable
                return False
            except Exception as e:
                eprint("Exception during evaluation:", e)
                return False

            for x, y in self.evalExamples:
                if self.cache and (x, e) in EVALUATIONTABLE:
                    p = EVALUATIONTABLE[(x, e)]
                else:
                    try:
                        p = self.predict(f, x)
                    except BaseException:
                        p = None
                    if self.cache:
                        EVALUATIONTABLE[(x, e)] = p
                if p != y:
                    if timeout is not None:
                        signal.signal(signal.SIGVTALRM, lambda *_: None)
                        signal.setitimer(signal.ITIMER_VIRTUAL, 0)
                    return False

            return True
        # except e:
            # eprint(e)
            # assert(False)
        except EvaluationTimeout:
            eprint("Timed out while evaluating", e)
            return False
        finally:
            if timeout is not None:
                signal.signal(signal.SIGVTALRM, lambda *_: None)
                signal.setitimer(signal.ITIMER_VIRTUAL, 0)

def retrieveARCJSONTasks(directory, filenames=None):

    # directory = '/Users/theo/Development/program_induction/ec/ARC/data/training'
    data = []

    for filename in os.listdir(directory):
        if ("json" in filename):
            task = retrieveARCJSONTask(filename, directory)
            if filenames is not None:
                if filename in filenames:
                    data.append(task)
            else:
                data.append(task)
    return data


def retrieveARCJSONTask(filename, directory):
    with open(directory + "/" + filename, "r") as f:
        loaded = json.load(f)

    ioExamples = [
            ((Grid(gridArray=example["input"]),), Grid(gridArray=example["output"]))
            for example in loaded["train"]
        ]
    evalExamples = [
            ((Grid(gridArray=example["input"]),), Grid(gridArray=example["output"]))
            for example in loaded["test"]
        ]

    task = ArcTask(
        filename,
        arrow(tgridin, tgridout),
        ioExamples,
        evalExamples
    )
    task.specialTask = ('arc', 5)
    return task


def list_options(parser):
    # parser.add_argument("--random-seed", type=int, default=17)
    # parser.add_argument("--train-few", default=False, action="store_true")
    parser.add_argument("--firstTimeEnumerationTimeout", type=int, default=3600)
    parser.add_argument("--featureExtractor", choices=[
        "dummy",
        "ArCNN",
        "ArcCnnEmbed"
        ])

    # parser.add_argument("-i", type=int, default=10)


def check(filename, f, directory):
    train, test = retrieveARCJSONTask(filename, directory=directory)
    print(train)

    for input, output in train.examples:
        input = input[0]
        if f(input) == output:
            print("HIT")
        else:
            print("MISS")
            print("Got")
            f(input).pprint()
            print("Expected")
            output.pprint()

    return


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def gridToArray(grid):
    temp = np.full((grid.getNumRows(),grid.getNumCols()),None)
    for yPos,xPos in grid.points:
        temp[yPos, xPos] = str(grid.points[(yPos,xPos)])
    return temp

class ArcCNN(nn.Module):
    special = 'arc'
    
    def __init__(self, tasks=[], testingTasks=[], cuda=False, H=64, inputDimensions=25):
        super(ArcCNN, self).__init__()
        self.CUDA = cuda
        self.recomputeTasks = True
        self.outputDimensionality = H

        if cuda:
            self.CUDA=True
            self.cuda()  # I think this should work?

        self.linear = nn.Linear(inputDimensions,H)
        # self.hidden = nn.Linear(H, H)

    def forward(self, v, v2=None):

        v = F.relu(self.linear(v))
        return v.view(-1)

    def featuresOfTask(self, t, t2=None):  # Take a task and returns [features]
        v = None
        for example in t.examples[-1:]:
            inputGrid, outputGrid = example
            inputGrid = inputGrid[0]

            inputVector = np.array(gridToArray(inputGrid)).flatten().astype(np.float32)
            paddedInputVector = nn.functional.pad(torch.from_numpy(inputVector), (0,900 - inputVector.shape[0]), 'constant', 0)

            outputVector = np.array(gridToArray(outputGrid)).flatten().astype(np.float32)
            paddedOutputVector = nn.functional.pad(torch.from_numpy(outputVector), (900 - outputVector.shape[0],0), 'constant', 0)

            exampleVector = torch.cat([paddedInputVector, paddedOutputVector], dim=0)
            if v is None:
                v = exampleVector
            else:
                v = torch.cat([v, exampleVector], dim=0)
        return self(v)

    def taskOfProgram(self, p, tp):
        """
        For simplicitly we only use one example per task randomly sampled from
        all possible input grids we've seen.
        """
        def randomInput(t): return random.choice(self.argumentsWithType[t])

        startTime = time.time()
        examples = []
        while True:
            # TIMEOUT! this must not be a very good program
            if time.time() - startTime > self.helmholtzTimeout: return None

            # Grab some random inputs
            xs = [randomInput(t) for t in tp.functionArguments()]
            try:
                y = runWithTimeout(lambda: p.runWithArguments(xs), self.helmholtzEvaluationTimeout)
                examples.append((tuple(xs),y))
                if len(examples) >= 1:
                    return Task("Helmholtz", tp, examples)
            except: continue
        return None

    # def customFeaturesOfTask(self, t):
    #     v = None
    #     for example in t.examples[-1:]:
    #         inputGrid, outputGrid = example
    #         inputGrid = inputGrid[0]

    #         inputColors, outputColors = set(inputGrid.points.values()), set(outputGrid.points.values())
    #         specialColorsInput = inputColors - outputColors
    #         specialColorsInputVector = [int(i in specialColorsInput) for i in range(10)]
    #         specialColorsOutput = outputColors - inputColors
    #         specialColorsOutputVector = [int(i in specialColorsOutput) for i in range(10)]
    #         changeDimensions = [int((inputGrid.getNumCols() != outputGrid.getNumCols()) or (inputGrid.getNumRows() != outputGrid.getNumRows()))]
    #         useSplitBlocks = [int(((inputGrid.getNumCols()//outputGrid.getNumCols()) == 2) or ((inputGrid.getNumRows()//outputGrid.getNumRows()) == 2))]
    #         fractionBlackBInput = [sum([c == 0 for c in inputGrid.points.values()]) / len(inputGrid.points)]
    #         fractionBlackBOutput = [sum([c == 0 for c in outputGrid.points.values()]) / len(outputGrid.points)]
    #         pixelWiseError = [0 if (changeDimensions[0] == 1) else (sum([outputGrid.points[key] == outputGrid.points[key] for key in outputGrid.points.keys()]) / len(outputGrid.points))]

    #         finalVector = np.array([specialColorsInputVector + specialColorsOutputVector + changeDimensions + useSplitBlocks + fractionBlackBInput + fractionBlackBOutput + pixelWiseError]).astype(np.float32)
    #         finalTensor = torch.from_numpy(finalVector)
    #         # print(finalTensor)
    #         if v is None:
    #             v = finalTensor
    #         else:
    #             v = torch.cat([v, finalTensor], dim=0)
    #     return self(v)


    def featuresOfTasks(self, ts, t2=None):  # Take a task and returns [features]
        """Takes the goal first; optionally also takes the current state second"""
        return [self.featuresOfTask(t) for t in ts]


def main(args):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on manipulating sequences of numbers.

    """
    # samples = {
    #     "007bbfb7.json": _solve007bbfb7,
    #     "c9e6f938.json": _solvec9e6f938,
    #     "50cb2852.json": lambda grid: _solve50cb2852(grid)(8),
    #     "fcb5c309.json": _solvefcb5c309,
    #     "97999447.json": _solve97999447,
    #     "f25fbde4.json": _solvef25fbde4,
    #     "72ca375d.json": _solve72ca375d,
    #     "5521c0d9.json": _solve5521c0d9,
    #     "ce4f8723.json": _solvece4f8723,
    # }

    import os

    homeDirectory = "/".join(os.path.abspath(__file__).split("/")[:-4])
    dataDirectory = homeDirectory + "/arc-data/data/"

    trainTasks = retrieveARCJSONTasks(dataDirectory + 'training', None)
    holdoutTasks = retrieveARCJSONTasks(dataDirectory + 'evaluation')

    baseGrammar = Grammar.uniform(basePrimitives() + leafPrimitives())
    # print("base Grammar {}".format(baseGrammar))

    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/arc/%s" % timestamp
    os.system("mkdir -p %s" % outputDirectory)

    args.update(
        {"outputPrefix": "%s/arc" % outputDirectory, "evaluationTimeout": 1,}
    )

    # # nnTrainTask, _ = retrieveARCJSONTasks(dataDirectory, ['dae9d2b5.json'])
    # # arcNN = ArcCNN(inputDimensions=25)
    # # v = arcNN.featuresOfTask(nnTrainTask[0])
    # # print(v)

    # print("homeDirectory: {}".format(homeDirectory))
    # resumeDirectory = '/experimentOutputs/arc/2020-04-27T15:41:52.288988/'
    # pickledFile = 'arc_aic=1.0_arity=3_ET=28800_t_zero=28800_it=1_MF=10_noConsolidation=True_pc=30.0_RW=False_solver=ocaml_STM=True_L=1.0_TRR=default_K=2_topkNotMAP=False_rec=False.pickle'
    # result, firstFrontier, allFrontiers, frontierOverTime, topDownGrammar, preConsolidationGrammar, resumeRecognizer, learnedProductions = getTrainFrontier(homeDirectory + resumeDirectory + pickledFile, 0)

    # print(topDownGrammar)
    # print("-----------------------------------------------------")

    # print(result.parameters)
    # for i,frontier in enumerate(firstFrontier):
    #     print(i, frontier.topK(1).entries[0].program)

    # timeout = 10.0
    featureExtractor = {
        "dummy": DummyFeatureExtractor,
    }[args.pop("featureExtractor")]

    # recognizer = RecognitionModel(featureExtractor, topDownGrammar)
    # request = arrow(tgridin, tgridout)
    # inputs = [inputGrid[0].toJson() for task in trainTasks for inputGrid, _ in task.examples]
    # print(inputs)    

    # helmholtzEnumeration(baseGrammar, request, inputs, timeout, _=None,
                         # special="arc", evaluationTimeout=None)
    # sample = recognizer.sampleHelmholtz([request], statusUpdate='.', seed=1)


    # print(sample)


    #     print(featureExtractor.featuresOfTask(task))
    # timeout = 1200
    # path = "recognitionModels/{}_trainTasks={}_timeout={}".format(datetime.datetime.now(), len(firstFrontier), timeout)
    
    # trainedRecognizer = sleep_recognition(None, baseGrammar, [], [], [], firstFrontier, featureExtractor=ArcCNN, activation='tanh', CPUs=1, timeout=timeout, helmholtzFrontiers=[], helmholtzRatio=0, solver='ocaml', enumerationTimeout=0, skipEnumeration=True)
    
    # with open(path,'wb') as handle:
    #     dill.dump(trainedRecognizer, handle)
    #     print('Stored recognizer at: {}'.format(path))


    # trainedRecognizerPath = 'recognitionModels/2020-04-26 15:05:28.972185_trainTasks=2343_timeout=1200'
    # with open(trainedRecognizerPath, 'rb') as handle:
    #     trainedRecognizer = dill.load(handle)

    explorationCompression(baseGrammar, trainTasks, featureExtractor=featureExtractor, testingTasks=[], **args)