from dreamcoder.domains.re2.makeRe2Tasks import loadRe2Dataset
from dreamcoder.dreamcoder import ecIterator
from dreamcoder.utilities import *
from dreamcoder.domains.text.textPrimitives import primitives
from dreamcoder.domains.list.listPrimitives import bootstrapTarget
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

def main(args):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on re2 tasks.
    """
    task_dataset = args["taskDataset"]
    task_dataset_dir=args.pop("taskDatasetDir")
    train, test = loadRe2Dataset(task_dataset=task_dataset, task_dataset_dir=task_dataset_dir)
    eprint(f"Loaded dataset [{task_dataset}]: [{len(train)}] train and [{len(test)}] test tasks.")
    
    baseGrammar = Grammar.uniform(primitives +  bootstrapTarget())
    
    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/re2/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)
    
    ## DEBUGGING
    train = [train[0]]
    train[0].examples = [
        (x, ['w'] +  y[:-1] )
        for (x, y) in train[0].examples
    ]
    
    
    evaluationTimeout = 0.0005
    
    generator = ecIterator(baseGrammar, train,
                           testingTasks=test,
                           outputPrefix="%s/re2"%outputDirectory,
                           evaluationTimeout=evaluationTimeout,
                           **args)
    for result in generator:
        pass