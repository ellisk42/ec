from dreamcoder.domains.re2.makeRe2Tasks import loadRe2Dataset
from dreamcoder.dreamcoder import ecIterator
from dreamcoder.utilities import *
from dreamcoder.domains.text.main import ConstantInstantiateVisitor
from dreamcoder.domains.text.textPrimitives import re2_primitives, primitives, re2_4_letter, re2_6_letter, re2_characters
from dreamcoder.domains.list.listPrimitives import re2_ListPrimitives, bootstrapTarget
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
                            "re2_500",
                            "re2_500_aesr",
                            "re2_500_aesdrt"],
                        default="re2_3000",
                        help="Load pre-generated task datasets.")
    parser.add_argument("--taskDatasetDir",
                        default="data/re2/tasks")
    parser.add_argument("--languageDatasetDir",
                        default="data/re2/language")
    parser.add_argument("--iterations_as_epochs",
                        default=True)
    parser.add_argument("--primitives",
                        choices=[
                            "re2_primitives",
                            "re2_4_letter",
                            "re2_6_letter"],
                        default="re2_primitives",
                        help="Which primitive set to use, which may restrict the number of characters we allow.")
    parser.add_argument("--allow_language_strings",
                        default=False)

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
    if use_epochs and args["taskBatchSize"] is not None:
        eprint("Using iterations as epochs")
        args["iterations"] *= int(len(train) / args["taskBatchSize"]) 
        eprint(f"Now running for n={args['iterations']} iterations.")
    
    which_prims = args.pop("primitives")
    if which_prims == 're2_primitives':
        primitives = re2_primitives
    elif which_prims == 're2_4_letter':
        primitives = re2_4_letter
    else:
        primitives = re2_6_letter
    
    baseGrammar = Grammar.uniform(primitives + re2_ListPrimitives())
    print("Using base grammar:")
    print(baseGrammar)

    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/re2/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)
    
    # TODO (@Cathy Wong): allow this to include sequences specified in the language.
    allow_language_strings = args.pop("allow_language_strings")
    eprint(f"Allowing language constants: [{allow_language_strings}]")
    if allow_language_strings:
        eprint("Not yet implemented!")
        assert False
    words = re2_characters
    ConstantInstantiateVisitor.SINGLE = \
        ConstantInstantiateVisitor(words)
    evaluationTimeout = 0.0005
    
    generator = ecIterator(baseGrammar, train,
                           testingTasks=test,
                           outputPrefix="%s/re2"%outputDirectory,
                           evaluationTimeout=evaluationTimeout,
                           **args)
    for result in generator:
        pass