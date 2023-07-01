#!/usr/bin/env python3
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from click import argument
from dreamcoder.domains.quantum_circuits.main import main
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs

import sys
sys.path.append("../")

import dreamcoder as dc

from dreamcoder.domains.quantum_circuits.primitives import primitives, grammar
from dreamcoder.domains.quantum_circuits.tasks import makeTasks

import os
import datetime

def quantum_extras(parser):
    parser.add_argument("--limitedConnectivity", action='store_true')
    parser.add_argument("--outputDirectory", default="default", type=str)
    parser.add_argument("--nqubit", default=3, type=int)
    parser.add_argument("--fromGrammar", default=None, type=str)


if __name__ == '__main__':

    arguments = commandlineArguments(
        CPUs=numberOfCPUs(),
        iterations=100,  # 40
        enumerationTimeout=200,  # 150,#-g  #1000
        taskBatchSize=25,  # smaller should be faster
        taskReranker="randomShuffle",  # default
        structurePenalty=6,  # increase regularization 3 4 (it awas 1) look at a few [1,15]
        pseudoCounts=10,  # increase 100 test a few values
        solver="bottom",
        compressor="pypy",
        useRecognitionModel=False,
        featureExtractor=None,  # it was TowerCNN
        helmholtzRatio=0.5,
        recognitionTimeout=3,
        a=3,
        topK=2,
        maximumFrontier=5,
        extras=quantum_extras)  # ocaml, python, pypy

    if arguments["resume"] is not None and arguments["resume"][-1] == "/":
        import glob
        filenames = glob.glob(f"{arguments['resume']}quantum_train_*.pickle")
        filenames.sort(key=lambda x: int(x.split("it=")[1].split("_")[0]))
        arguments["resume"] = filenames[-1]
    main(arguments)
