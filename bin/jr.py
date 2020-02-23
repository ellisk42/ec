# experiments for Josh Rule
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.domains.list.main import main, list_options
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs

import sys
import random
from collections import defaultdict
import json
import math
import os
import datetime

from dreamcoder.dreamcoder import explorationCompression
from dreamcoder.utilities import eprint, flatten, testTrainSplit
from dreamcoder.grammar import Grammar
from dreamcoder.task import Task
from dreamcoder.enumeration import multicoreEnumeration
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
from dreamcoder.domains.list.listPrimitives import basePrimitives, primitives, McCarthyPrimitives, bootstrapTarget_extra, no_length, josh_primitives
from dreamcoder.recognition import RecurrentFeatureExtractor
from dreamcoder.domains.list.makeListTasks import make_list_bootstrap_tasks, sortBootstrap, EASYLISTTASKS, joshTasks


"""

z
Task: learn the 21 functions from wave 1 using the wave 1 DSL using the following simulation structure. Please follow this structure closely---it matches the human experiment against which we'll be comparing each model:
- For i_run in [0,1,2,...,9]:
  - create an empty training set.
  - For i_N in [0,1,2,...,19]:
    - Uniformly sample 1 test example not in training set.
    - Run/train your model with the training examples.
    - Given the hypothesis with the best score on the training data, predict the output of the test example given the input.
    - Record i_Run, i_N, concept id, test input, test output, prediction, and whether the prediction == test output.
    - Add the current test example to the training set.

Deliverables:
1. a CSV containing the record for each run as per above
2. any relevant notes on how you setup the simulation
3. a description of any relevant computational budget (e.g. number of programs sampled to train the network, number of programs considered during search before finding a correct program, total CPU/GPU time required to run model, number of resolutions, etc.)

If you have questions or need a different delivery date, please let me know.

Thanks!
Josh
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("--w","-w",default=1,type=int)
    parser.add_argument("--timeout","-t",default=600,type=float)
    parser.add_argument("--CPUs",default=numberOfCPUs(),type=int)
    arguments = parser.parse_args()

    tasks = joshTasks(arguments.w)
    

    timeout = arguments.timeout

    if arguments.w == 3:
        gs = [Grammar.uniform(pt)
              for pt in josh_primitives(arguments.w) ]
    else:
        g = Grammar.uniform(josh_primitives(arguments.w))
    for t in tasks:
        t.allExamples = t.examples
        t.testingExamples = t.examples
        t.examples = []

    tasks = tasks

    if arguments.w == 3:
        taskToGrammar = {t: gs[int(int(t.name.split("_")[0]) >= 81)]
                         for t in tasks }
        for trial in range(11):
            frontiers, times, pcs = multicoreEnumeration(taskToGrammar,tasks,solver="ocaml",maximumFrontier=1,
                                                         enumerationTimeout=timeout,CPUs=arguments.CPUs,
                                                         evaluationTimeout=0.001,
                                                         testing=True)
            frontiers = {f.task: f for f in frontiers}
            eprint("evaluating on held out examples")
            for ti,t in enumerate(tasks):
                testingExample = t.allExamples[trial]
                eprint(t)
                eprint("trained on")
                eprint(t.examples)
                eprint("evaluated on")
                eprint(testingExample)
                X = str(testingExample[0][0]).replace(",","").replace("[","(").replace("]",")")
                Y = str(testingExample[1]).replace(",","").replace("[","(").replace("]",")")

                if len(frontiers[t]):
                    p = frontiers[t].topK(1).entries[0].program
                    eprint("best program",p)
                    try:
                        yh = p.evaluate([])(testingExample[0][0])
                    except: yh = ""
                    eprint("gives the prediction")
                    eprint(yh)
                    correct = yh == testingExample[1]
                    eprint("is this correct?",correct)
                    yh = str(yh).replace(",","").replace("[","(").replace("]",")")

                    p = str(p).replace("fix1","fix").replace("gt?",">").replace("-n99","-").replace("-n9","-").replace("+n99","+").replace("+n9","+").replace("car","head").replace("cdr","tail").replace("empty?","is_empty").replace("eq?","is_equal")
                    print(f"CSVc{t.name.split('_')[0]},1,{t.name.split('_')[1]},{trial},\"{p}\",{times[t]},{pcs[t]}")
                else:
                    print(f"CSVc{t.name.split('_')[0]},1,{t.name.split('_')[1]},{trial},\"(lambda $0)\",0,1")
                    eprint("could not find any program")

                t.examples.append(testingExample)
            
        sys.exit(0)

    for r in range(10):
        # create empty training set
        for t in tasks: t.examples = []

        for n in range(20):
            frontiers, times, pcs = multicoreEnumeration(g,tasks,solver="ocaml",maximumFrontier=1,
                                                         enumerationTimeout=timeout,CPUs=arguments.CPUs,
                                                         evaluationTimeout=0.001,
                                                         testing=True)
            frontiers = {f.task: f for f in frontiers}
            eprint("evaluating on held out examples")
            for ti,t in enumerate(tasks):
                testingExample = random.choice([e for e in t.allExamples
                                                if e not in t.examples])
                eprint(t)
                eprint("trained on")
                eprint(t.examples)
                eprint("evaluated on")
                eprint(testingExample)
                X = str(testingExample[0][0]).replace(",","").replace("[","(").replace("]",")")
                Y = str(testingExample[1]).replace(",","").replace("[","(").replace("]",")")

                if len(frontiers[t]):
                    p = frontiers[t].topK(1).entries[0].program
                    eprint("best program",p)
                    try:
                        yh = p.evaluate([])(testingExample[0][0])
                    except: yh = ""
                    eprint("gives the prediction")
                    eprint(yh)
                    correct = yh == testingExample[1]
                    eprint("is this correct?",correct)
                    yh = str(yh).replace(",","").replace("[","(").replace("]",")")

                    print(f"CSV{r},{n},{ti},{X},{Y},{yh},{int(correct)}")
                else:
                    eprint("could not find any program")
                    print(f"CSV{r},{n},{ti},{X},{Y},,0")

                t.examples.append(testingExample)





