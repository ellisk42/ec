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
from dreamcoder.recognition import RecognitionModel, DummyFeatureExtractor, variable
from dreamcoder.program import Program
from dreamcoder.domains.arc.utilsPostProcessing import *
from dreamcoder.domains.arc.arcPrimitives import *
from dreamcoder.domains.arc.taskGeneration import *

DEFAULT_DOMAIN_NAME_PREFIX = "arc"
DEFAULT_LANGUAGE_DATASET_DIR = f"data/{DEFAULT_DOMAIN_NAME_PREFIX}/language"

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


def arc_options(parser):
    # parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--unigramEnumerationTimeout", type=int, default=3600)
    parser.add_argument("--firstTimeEnumerationTimeout", type=int, default=1)
    parser.add_argument("--featureExtractor", default="dummy", choices=[
        "arcCNN",
        "dummy"])

    parser.add_argument("--languageDatasetDir",             
        default=DEFAULT_LANGUAGE_DATASET_DIR,
        help="Top-level directory containing the language tasks.")


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
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        self.gridDimension = 30

        # channels for hidden
        hid_dim = 64
        z_dim = 64

        self.encoder = nn.Sequential(
            conv_block(22, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            Flatten()
        )

    def forward(self, v):
        """ """
        assert v.shape == (v.shape[0], 22, self.gridDimension, self.gridDimension)
        v = variable(v, cuda=self.CUDA).float()
        v = self.encoder(v)
        return v.mean(dim=0)


    def featuresOfTask(self, t):  # Take a task and returns [features]
        v = None
        for example in t.examples:
            inputGrid, outputGrid = example
            inputGrid = inputGrid[0]

            inputTensor = inputGrid.to_tensor(grid_height=30, grid_width=30)
            outputTensor = outputGrid.to_tensor(grid_height=30, grid_width=30)
            ioTensor = torch.cat([inputTensor, outputTensor], 0).unsqueeze(0)

            if v is None:
                v = ioTensor
            else:
                v = torch.cat([v, ioTensor], dim=0)
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
    dataDirectory = homeDirectory + "/arc_data/data/"


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

    featureExtractor = {
        "dummy": DummyFeatureExtractor,
        "arcCNN": ArcCNN
    }[args.pop("featureExtractor")]

    explorationCompression(baseGrammar, trainTasks, featureExtractor=featureExtractor, testingTasks=[], **args)
