from dreamcoder.domains.re2.makeRe2Tasks import loadRe2Dataset, buildRE2MockTask
from dreamcoder.dreamcoder import ecIterator
from dreamcoder.utilities import *
from dreamcoder.domains.text.main import ConstantInstantiateVisitor
from dreamcoder.domains.re2.re2Primitives import *
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
                        nargs="*",
                        default=["re2_chars_None", "re2_bootstrap_v1_primitives", "re2_bootstrap_list_primitives"],
                        help="Which primitive set to use, which may restrict the number of characters we allow.")
    parser.add_argument("--allow_language_strings",
                        default=False)
    parser.add_argument("--run_python_test",
                        action='store_true')
    parser.add_argument("--run_ocaml_test",
                        action='store_true')

def main(args):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on re2 tasks.
    """
    primitive_names = args.pop("primitives")
    primitives, type_request = load_re2_primitives(primitive_names)
    
    baseGrammar = Grammar.uniform(primitives)
    print("Using base grammar:")
    print(baseGrammar)
    
    task_dataset = args["taskDataset"]
    task_dataset_dir=args.pop("taskDatasetDir")
    train, test = loadRe2Dataset(task_dataset=task_dataset, task_dataset_dir=task_dataset_dir, type_request=type_request)
    eprint(f"Loaded dataset [{task_dataset}]: [{len(train)}] train and [{len(test)}] test tasks.")
    
    if args.pop("run_python_test"):
        re2_primitives_main()
        assert False
    if args.pop("run_ocaml_test"):
        tasks = [buildRE2MockTask(train[0])]
        if False:
            # Tests the Helmholtz enumeration.
            from dreamcoder.dreaming import backgroundHelmholtzEnumeration
            helmholtzFrontiers = backgroundHelmholtzEnumeration(tasks, 
                                                                baseGrammar, 
                                                                timeout=1,
                                                                evaluationTimeout=10,
                                                                special=None,
                                                                executable='re2Test',
                                                                serialize_special=None)
            f = helmholtzFrontiers()
        import pdb; pdb.set_trace()
        assert False
    
    use_epochs = args.pop("iterations_as_epochs")
    if use_epochs and args["taskBatchSize"] is not None:
        eprint("Using iterations as epochs")
        args["iterations"] *= int(len(train) / args["taskBatchSize"]) 
        eprint(f"Now running for n={args['iterations']} iterations.")

    timestamp = datetime.datetime.now().isoformat()
    # Escape the timestamp.
    timestamp = timestamp.replace(":", "-")
    timestamp = timestamp.replace(".", "-")
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
    evaluationTimeout = 0.05
    
    generator = ecIterator(baseGrammar, train,
                           testingTasks=test,
                           outputPrefix="%s/re2"%outputDirectory,
                           evaluationTimeout=evaluationTimeout,
                           **args)
    for result in generator:
        pass