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
    import dill as pickle
except: pass


def main(arguments):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on a set of quantum-algorithm tasks.
    """   
    dc.domains.quantum_circuits.primitives.GLOBAL_LIMITED_CONNECTIVITY = False
    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/quantum/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)
    
    tasks = makeTasks(30) #15
    train_tasks, test_tasks = train_test_split(tasks, test_size=0.5)

    # # check LIMITED_CONNECTIVITY
    if arguments["limited_connectivity"]: 
        dc.domains.quantum_circuits.primitives.GLOBAL_LIMITED_CONNECTIVITY = True
        dc.utilities.eprint("Limited qubit connectivity enforced")
    del arguments["limited_connectivity"]
    
    # # TRAIN
    g0 = grammar
    generator = dc.dreamcoder.ecIterator(g0, train_tasks,
                           testingTasks=[],
                           outputPrefix=f"{outputDirectory}/quantum_train",
                        #    evaluationTimeout=10,
                           **arguments)
    for result in generator: ...
    
    
    arguments["noConsolidation"]=True
    arguments["iterations"]=1
    del arguments["taskBatchSize"]
    del arguments["taskReranker"]
    
    # # # TEST
    # final grammar
    g0 = result.grammars[-1]
    # or load from file
    # with open("experimentOutputs/quantum/long_grammar","rb") as f:
    #     g0 = pickle.load(f)
    # with open("experimentOutputs/quantum/long_test_tasks","rb") as f:
    #     test_tasks = pickle.load(f)
    
    generator = dc.dreamcoder.ecIterator(g0, test_tasks,
                        testingTasks=[],
                        outputPrefix=f"{outputDirectory}/quantum_test",
                        **arguments)
    for result in generator: ...


    # FULL GRAMMAR on TEST
    dc.domains.quantum_circuits.primitives.GLOBAL_LIMITED_CONNECTIVITY = False
    generator = dc.dreamcoder.ecIterator(full_grammar, test_tasks,
                    testingTasks=[],
                    outputPrefix=f"{outputDirectory}/quantum_full",
                    **arguments)
    for result in generator: ...
    
    return outputDirectory

