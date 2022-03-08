import sys
sys.path.append("../")

import dreamcoder as dc

from dreamcoder.domains.quantum_algorithms.primitives import primitives, grammar
from dreamcoder.domains.quantum_algorithms.tasks import makeTasks

import os
import datetime


try: #pypy will fail
    from dreamcoder.recognition import variable
    import torch.nn as nn
    import torch.nn.functional as F

except: pass


def main(arguments):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on a set of quantum-algorithm tasks.
    """   
    global primitives
    
    g0 = grammar
    tasks = makeTasks()
    
    # what about checkpoints
    
    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/quantum/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)
    
    evaluationTimeout = 0.01
    generator = dc.dreamcoder.ecIterator(g0, tasks,
                           testingTasks=[],
                           outputPrefix="%s/quantum"%outputDirectory,
                           evaluationTimeout=evaluationTimeout,
                           **arguments)
    for result in generator:
        continue
    
    pass