from dreamcoder.domains.re2.makeRe2Tasks import loadRe2Dataset
from dreamcoder.dreamcoder import ecIterator
from dreamcoder.utilities import *
from dreamcoder.domains.text.main import ConstantInstantiateVisitor
from dreamcoder.domains.text.textPrimitives import re2_primitives
from dreamcoder.domains.list.listPrimitives import re2_ListPrimitives
from dreamcoder.recognition import *
from dreamcoder.enumeration import *

import os
import datetime
import random

def re2_options(parser):
    parser.add_argument("--taskDataset", type=str,
                        choices=[
                            "re2_3000",
                            "re2_1000",
                            "re2_500"],
                        default="re2_3000",
                        help="Load pre-generated task datasets.")
    parser.add_argument("--taskDatasetDir",
                        default="data/re2/tasks")
    parser.add_argument("--languageDatasetDir",
                        default="data/re2/language")
    parser.add_argument("--iterations_as_epochs",
                        default=True)

def main(args):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on re2 tasks.
    """
    task_dataset = args["taskDataset"]
    task_dataset_dir=args.pop("taskDatasetDir")
    train, test = loadRe2Dataset(task_dataset=task_dataset, task_dataset_dir=task_dataset_dir)
    eprint(f"Loaded dataset [{task_dataset}]: [{len(train)}] train and [{len(test)}] test tasks.")

    use_epochs = args.pop("iterations_as_epochs")
    if use_epochs:
        eprint("Using iterations as epochs")
        args["iterations"] *= int(len(train) / args["taskBatchSize"]) 
        eprint(f"Now running for n={args['iterations']} iterations.")
    
    baseGrammar = Grammar.uniform(re2_primitives +  re2_ListPrimitives())
    
    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/re2/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)
    
    ConstantInstantiateVisitor.SINGLE = \
        ConstantInstantiateVisitor(list(map(list, list({tuple([c for c in s])
                                                        for t in test + train
                                                        for s in t.stringConstants}))))
    evaluationTimeout = 0.0005
    
    generator = ecIterator(baseGrammar, train,
                           testingTasks=test,
                           outputPrefix="%s/re2"%outputDirectory,
                           evaluationTimeout=evaluationTimeout,
                           **args)
    for result in generator:
        pass