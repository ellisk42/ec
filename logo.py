from ec import explorationCompression, commandlineArguments
from grammar import Grammar
from utilities import eprint, testTrainSplit, numberOfCPUs
from makeLogoTasks import makeTasks
from logoPrimitives import primitives, turtle
from math import log
from collections import OrderedDict
from program import Program
from task import Task

import random as random
import json
import torch
import png
import time
import subprocess
import os
import torch.nn as nn

from recognition import variable

global prefix_dreams


def list_options(parser):
    parser.add_argument("--target", type=str,
                        default=[],
                        action='append',
                        help="Which tasks should this try to solve")
    parser.add_argument("--reduce", type=str,
                        default=[],
                        action='append',
                        help="Which tasks should this try to solve")
    parser.add_argument("--save", type=str,
                        default=None,
                        help="Filepath output the grammar if this is a child")
    parser.add_argument("--prefix", type=str,
                        default="experimentOutputs/geom",
                        help="Filepath output the grammar if this is a child")


if __name__ == "__main__":
    args = commandlineArguments(
        steps=1000,
        a=3,
        topK=5,
        iterations=10,
        useRecognitionModel=True,
        helmholtzRatio=0.5,
        helmholtzBatch=500,
        maximumFrontier=1000,
        CPUs=numberOfCPUs(),
        pseudoCounts=10.0,
        activation="tanh",
        extras=list_options)
    target = args.pop("target")
    red = args.pop("reduce")
    save = args.pop("save")
    prefix = args.pop("prefix")
    prefix_dreams = prefix + "/dreams/" + ('_'.join(target)) + "/"
    prefix_pickles = prefix + "/pickles/" + ('_'.join(target)) + "/"
    if not os.path.exists(prefix_dreams):
        os.makedirs(prefix_dreams)
    if not os.path.exists(prefix_pickles):
        os.makedirs(prefix_pickles)
    tasks = makeTasks(target)
    eprint("Generated", len(tasks), "tasks")

    test, train = testTrainSplit(tasks, 1.)
    eprint("Split tasks into %d/%d test/train" % (len(test), len(train)))

    if red is not []:
        for reducing in red:
            try:
                with open(reducing, 'r') as f:
                    prods = json.load(f)
                    for e in prods:
                        e = Program.parse(e)
                        if e.isInvented:
                            primitives.append(e)
            except EOFError:
                eprint("Couldn't grab frontier from " + reducing)
            except IOError:
                eprint("Couldn't grab frontier from " + reducing)
            except json.decoder.JSONDecodeError:
                eprint("Couldn't grab frontier from " + reducing)

    primitives = list(OrderedDict((x, True) for x in primitives).keys())
    baseGrammar = Grammar.uniform(primitives)

    eprint(baseGrammar)

    r = explorationCompression(baseGrammar, train,
                               testingTasks=test,
                               outputPrefix=prefix_pickles,
                               compressor="rust",
                               evaluationTimeout=0.01,
                               **args)
    needsExport = [str(z)
                   for _, _, z
                   in r.grammars[-1].productions
                   if z.isInvented]
    if save is not None:
        with open(save, 'w') as f:
            json.dump(needsExport, f)

