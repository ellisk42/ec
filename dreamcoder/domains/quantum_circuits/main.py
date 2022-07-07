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
    import numpy as np
except: pass


def get_tasks(enumeration_timeout, label):
    if os.path.exists(f"experimentOutputs/quantum/{label}_train_tasks"):
        dc.utilities.eprint(f"Loading task dataset {label}.")
        with open(f"experimentOutputs/quantum/{label}_train_tasks","rb") as f:
            train_tasks=pickle.load(f)

        with open(f"experimentOutputs/quantum/{label}_test_tasks","rb") as f:
            test_tasks=pickle.load(f)
        
        with open(f"experimentOutputs/quantum/{label}_tasks","rb") as f:
            tasks=pickle.load(f)
        
    else: 
        dc.utilities.eprint("No task dataset found, generating a new one.")
        
        tasks = makeTasks(enumeration_timeout) 
        n_train = int(len(tasks)/20)
        
        total_indices= np.arange(len(tasks))
        probs = np.array([task.name[6:].count("(") for task in tasks],dtype=float)
        for i in np.arange(np.max(probs)+1):
            m = probs[probs==i].sum()
            probs[probs==i]/=m
        weight=probs/probs.sum()
        
        indices = set(np.random.choice(total_indices, n_train,p=weight))
        # remaining_indices = set(total_indices) - indices
        
        train_tasks = [tasks[i] for i in indices]
        test_tasks = [task for task in tasks if task not in train_tasks]
        
        with open(f"experimentOutputs/quantum/{label}_train_tasks","wb") as f:
            pickle.dump(train_tasks,f)

        with open(f"experimentOutputs/quantum/{label}_test_tasks","wb") as f:
            pickle.dump(test_tasks,f)
        
        with open(f"experimentOutputs/quantum/{label}_tasks","wb") as f:
            pickle.dump(tasks,f)
        
    return tasks, train_tasks, test_tasks


def test_grammar(grammar, tasks, arguments, path_and_label):
    arguments= arguments.copy()
    dc.utilities.eprint(f"Testing extracted grammar on {len(tasks)} tasks.")
    arguments["noConsolidation"]=True
    arguments["iterations"]=1
    del arguments["taskBatchSize"]
    del arguments["taskReranker"]
    del arguments["resume"]
    
    # or load from file
    # with open("experimentOutputs/quantum/long_grammar","rb") as f:
    #     g0 = pickle.load(f)
    
    generator = dc.dreamcoder.ecIterator(grammar, tasks,
                        testingTasks=[],
                        outputPrefix=path_and_label,
                        **arguments)
    for result in generator: ...

def main(arguments):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on a set of quantum-algorithm tasks.
    """   
    initial_arguments = arguments.copy()
    command_string = sys.argv[0]
    dc.domains.quantum_circuits.primitives.GLOBAL_LIMITED_CONNECTIVITY = False
    timestamp = datetime.datetime.now().isoformat()
    
    if arguments["resume"] is None: 
        outputDirectory = "experimentOutputs/quantum/%s"%timestamp
    else: outputDirectory = "/".join(arguments["resume"].split("/")[:-1])
    del arguments["outputDirectory"]
    os.system("mkdir -p %s"%outputDirectory)
    
    # Get quantum task dataset
    tasks, train_tasks, test_tasks = get_tasks(15, "short")


    # check LIMITED_CONNECTIVITY
    if arguments["limitedConnectivity"]: 
        dc.domains.quantum_circuits.primitives.GLOBAL_LIMITED_CONNECTIVITY = True
        dc.utilities.eprint("Limited qubit connectivity enforced")
    del arguments["limitedConnectivity"]
    

    # TRAIN
    g0 = grammar
    generator = dc.dreamcoder.ecIterator(g0, train_tasks,
                           testingTasks=[],
                           outputPrefix=f"{outputDirectory}/quantum_train",
                           **arguments)
    for result in generator:          
        # TEST at each step
        test_grammar(result.grammars[-1], 
                     test_tasks, 
                     arguments, 
                     f"{outputDirectory}/quantum_test_{len(result.grammars)-1}")


    # FULL GRAMMAR on TEST
    dc.domains.quantum_circuits.primitives.GLOBAL_LIMITED_CONNECTIVITY = False
    test_grammar(full_grammar, test_tasks, arguments, f"{outputDirectory}/quantum_full")

    return outputDirectory
