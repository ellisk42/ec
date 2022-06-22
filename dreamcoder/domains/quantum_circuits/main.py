import sys
sys.path.append("../")

import dreamcoder as dc

from dreamcoder.domains.quantum_circuits.primitives import primitives, grammar, full_grammar
from dreamcoder.domains.quantum_circuits.tasks import QuantumTask, makeTasks

import os
import datetime


try: #pypy will fail
    from dreamcoder.recognition import variable
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.model_selection import train_test_split
except: pass


def main(arguments):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on a set of quantum-algorithm tasks.
    """   
    global primitives
    
    g0 = grammar
    tasks = makeTasks(4) #15
    
    # what about checkpoints
    
    train_tasks, test_tasks = train_test_split(tasks, test_size=0.5)
    
    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/quantum/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)
    
    # # evaluationTimeout = 10 # at the moment it is disabled
    # TRAIN
    generator = dc.dreamcoder.ecIterator(g0, train_tasks,
                           testingTasks=[],
                           outputPrefix=f"{outputDirectory}/quantum_train",
                        #    evaluationTimeout=evaluationTimeout,
                           **arguments)
    for result in generator: ...
    
    # # # TEST
    g0 = result.grammars[-1]
    arguments["noConsolidation"]=True
    arguments["iterations"]=1
    del arguments["taskBatchSize"]
    del arguments["taskReranker"]
    generator = dc.dreamcoder.ecIterator(g0, test_tasks,
                        testingTasks=[],
                        outputPrefix=f"{outputDirectory}/quantum_test",
                        **arguments)
    for result in generator: ...

    # FULL GRAMMAR on TEST
    generator = dc.dreamcoder.ecIterator(full_grammar, test_tasks,
                    testingTasks=[],
                    outputPrefix=f"{outputDirectory}/quantum_full",
                    **arguments)
    for result in generator: ...
    
    return outputDirectory

