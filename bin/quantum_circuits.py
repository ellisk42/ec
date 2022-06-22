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

def quantum_extras(parser):
    parser.add_argument("--limited-connectivity",action='store_true')

if __name__ == '__main__': 
    arguments = commandlineArguments(
        featureExtractor=None, # it was TowerCNN
        CPUs=1,#numberOfCPUs(),
        helmholtzRatio=0.5,
        recognitionTimeout=3,
        iterations=10,#40
        a=3,
        structurePenalty=6, # increase regularization 3 4 (it was 1) look at a few [1,15]
        pseudoCounts=10,
        topK=2,
        maximumFrontier=5,
        solver="bottom", 
        useRecognitionModel=False,
        enumerationTimeout=10,#-g  #1000
        taskBatchSize=200,
        taskReranker="randomShuffle", #defualt
        compressor="pypy",
        extras=quantum_extras)   #ocaml, python, pypy  
    main(arguments)
    
    

    # resume the checkpoint
# -g (no neural network
# --solver python  (or pypy)

# python bin/quantum_algorithms.py -t 5 --compressor=pypy