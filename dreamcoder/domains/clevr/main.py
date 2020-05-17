from dreamcoder.utilities import *
from dreamcoder.recognition import *
from dreamcoder.enumeration import *
from dreamcoder.dreamcoder import ecIterator, default_wake_generative

from dreamcoder.domains.clevr.clevrPrimitives import *
from dreamcoder.domains.clevr.makeClevrTasks import *

import os
import datetime
import random

all_train_questions = [
    "1_zero_hop",
    '1_one_hop',
    '1_compare_integer',
    '1_same_relate',
    '1_single_or',
    '2_remove',
    '2_transform'
]

def clevr_options(parser):
    # Dataset loading options.
    parser.add_argument("--curriculumDatasets", type=str, nargs="*",
                        default=["curriculum"],
                        help="A list of curriculum datasets, stored as JSON CLEVR question files. These will be used in ")
    parser.add_argument("--taskDatasets", type=str, nargs="+",
                        default=all_train_questions,
                        help="Which task datasets to load, stored as JSON CLEVR question files.")
    parser.add_argument("--taskDatasetDir",
                        default="../too_clevr/data/clevr_dreams/",
                        help="Top level directory for the dataset.")
    parser.add_argument("--trainInputScenes",
                        default="CLEVR_train_scenes_1000",
                        help="Input scene graphs for all of the training questions.")
    parser.add_argument("--testInputScenes",
                        default="CLEVR_val_scenes_500",
                        help="Input scene graphs for all of the test questions.")

    # Primitive loading options.
    parser.add_argument("--primitives",
                        nargs="*",
                        default=["clevr_bootstrap", "clevr_map_transform"],
                        help="Which primitives to use. Choose from: [clevr_original, clevr_bootstrap, clevr_map_transform, clevr_filter, clevr_filter_except, clevr_difference]")
    parser.add_argument("--run_python_test",
                        action='store_true')
    parser.add_argument("--generate_ocaml_definitions",
                        action='store_true')
    parser.add_argument("--run_ocaml_test",
                        action='store_true')
                        
def main(args):
    # Load the curriculum and datasets.
    curriculum_datasets = args.pop("curriculumDatasets")
    task_dataset_dir=args.pop("taskDatasetDir")
    train_scenes, test_scenes = args.pop("trainInputScenes"), args.pop("testInputScenes")
    
    if len(curriculum_datasets) > 0:
        curriculum, _ = loadCLEVRDataset(task_datasets=curriculum_datasets, task_dataset_dir=task_dataset_dir, train_scenes=train_scenes, test_scenes = test_scenes, seed=args["seed"], is_curriculum=True)
    
    task_datasets = args["taskDatasets"]
    train, test = loadCLEVRDataset(task_datasets=task_datasets, task_dataset_dir=task_dataset_dir, train_scenes=train_scenes, test_scenes = test_scenes, seed=args["seed"])
    eprint(f"Loaded datasets: [{task_datasets}]: [{len(train)}] total train and [{len(test)}] total test tasks.")
    
    # Load the primitives and optionally run tests with the primitive set.
    primitive_names = args.pop("primitives")
    primitives = load_clevr_primitives(primitive_names)
    baseGrammar = Grammar.uniform(primitives)
    
    if args.pop("run_python_test"):
        run_clevr_primitives_test(primitive_names, curriculum)
        assert False
    
    if args.pop("generate_ocaml_definitions"):
        generate_ocaml_definitions(primitive_names)
        assert False
    
    if args.pop("run_ocaml_test"):
        # Test the Helmholtz enumeratio n
        # tasks = [buildClevrMockTask(train[0])]
        tasks = train
        if False:
            from dreamcoder.dreaming import backgroundHelmholtzEnumeration
            print(baseGrammar)
            helmholtzFrontiers = backgroundHelmholtzEnumeration(tasks, 
                                                                baseGrammar, 
                                                                timeout=10,
                                                                evaluationTimeout=0.05,
                                                                special='clevr',
                                                                executable='helmholtz',
                                                                serialize_special=serialize_clevr_object,
                                                                maximum_size=20000) # TODO: check if we need special to check tasks later
            f = helmholtzFrontiers()
        if True:
            # Check enumeration.
            tasks = [buildClevrMockTask(train[0])]
            default_wake_generative(baseGrammar, tasks, 
                                maximumFrontier=5,
                                enumerationTimeout=10,
                                CPUs=1,
                                solver='ocaml',
                                evaluationTimeout=0.05)
            
        