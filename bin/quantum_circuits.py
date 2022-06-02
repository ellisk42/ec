#!/usr/bin/env python3
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

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



if __name__ == '__main__': 
    arguments = commandlineArguments(
        featureExtractor=None, # it was TowerCNN
        CPUs=numberOfCPUs(),
        helmholtzRatio=0.5,
        recognitionTimeout=3,
        iterations=10,
        a=3,
        structurePenalty=1,
        pseudoCounts=10,
        topK=2,
        maximumFrontier=5,
        extras=None,
        solver="bottom", 
        useRecognitionModel=False,
        enumerationTimeout=300,#-g
        taskBatchSize=20,
        taskReranker="randomShuffle", #defualt
        compressor="pypy")   #ocaml, python, pypy  
    main(arguments)
    
    # resume the checkpoint
    # test on all the task set (which we should also save)

# -g (no neural network
# --solver python  (or pypy)

# python bin/quantum_algorithms.py -t 5 --compressor=pypy
## TASKS