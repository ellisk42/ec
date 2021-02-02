from dreamcoder.utilities import *
from dreamcoder.recognition import *
from dreamcoder.enumeration import *
from dreamcoder.dreamcoder import ecIterator, default_wake_generative

from dreamcoder.domains.clevr.clevrPrimitives import *
import dreamcoder.domains.clevr.cogsci_utils as cogsci_utils
import dreamcoder.domains.clevr.makeClevrTasks as makeClevrTasks
import dreamcoder.domains.clevr.test_makeClevrTasks as test_makeClevrTasks
import dreamcoder.domains.clevr.test_clevrPrimitives as test_clevrPrimitives
import dreamcoder.domains.clevr.test_clevrPrimitivesOcaml as test_clevrPrimitivesOcaml
import dreamcoder.domains.clevr.test_clevrRecognition as test_clevrRecognition
import dreamcoder.domains.clevr.test_clevrIntegration as test_clevrIntegration
import dreamcoder.domains.clevr.test_cogsci_utils as test_cogsci_utils
import dreamcoder.domains.clevr.test_makeClevrCogsciCurriculum as test_makeClevrCogsciCurriculum

import os
import random
"""
main_cogsci.py | Author : Catherine Wong.
This is the mian file for the CLEVR-based symbolic scene reasoning domain as it ran on Cogsci 2021. The primary difference is special logic for managing the training schedule: we configure this file to run a number of runs all at once in the same file, with the same conditions.
"""
DEFAULT_CLEVR_EVALUATION_TIMEOUT = 0.5
DEFAULT_CLEVR_DOMAIN_NAME_PREFIX = "clevr"
DEFAULT_COGSCI_PREFIX = "clevr_cogsci"
DEFAULT_TASK_DATASET_DIR = f"data/{DEFAULT_COGSCI_PREFIX}"
DEFAULT_LANGUAGE_DIR = f"data/{DEFAULT_COGSCI_PREFIX}/language/"

DEFAULT_EASY_CATEGORIES = ["2_localization", "1_zero_hop_no_string"]
DEFAULT_HARD_CATEGORIES = ["1_compare_integer_long", "2_remove_easy", "1_single_or_easy"]
DEFAULT_NUM_TRAIN_TASKS = [10, 5, 5]
DEFAULT_NUM_TEST_TASKS = [5, 5, 5]

def clevr_options(parser):
    # Experiment iteration parameters.
    parser.add_argument("--primitives",
                        nargs="*",
                        default=["clevr_bootstrap", "clevr_map_transform"],
                        help="Which primitives to use. Choose from: [clevr_original, clevr_bootstrap, clevr_map_transform, clevr_filter, clevr_filter_except, clevr_difference]")
    
    parser.add_argument("--easy_curriculum_categories",
                        nargs="*",
                        default=DEFAULT_EASY_CATEGORIES,
                        help="Which 'easy' categories to draw from at the start of the curriculum.")
    parser.add_argument("--hard_curriculum_categories",
                        nargs="*",
                        default=DEFAULT_HARD_CATEGORIES,
                        help="Which 'hard' categories to draw from at the start of the curriculum.")
    parser.add_argument("--num_train_tasks_per_category",
                        nargs="*",
                        default=DEFAULT_NUM_TRAIN_TASKS,
                        help="How many train tasks to see at each iteration.")
    parser.add_argument("--num_test_tasks_per_category",
                        nargs="*",
                        default=DEFAULT_NUM_TEST_TASKS,
                        help="Which testing tasks to see at each iteration.")
    parser.add_argument("--num_data_blocks_per_category",
                        required=True,
                        help="How many data blocks to run for each category.")
    
    
                        
    parser.add_argument("--evaluationTimeout",
                        default=DEFAULT_CLEVR_EVALUATION_TIMEOUT,
                        help="How long to spend evaluating a given CLEVR tasks.")
    parser.add_argument("--iterations_as_epochs",
                        default=True,
                        help="Whether to take the iterations value as an epochs value.")
    
    parser.add_argument("--run_cogsci_utils_test",
                        action='store_true',
                        help='Runs tests for cogsci utils methods for loading conditions.')      
    parser.add_argument("--run_makeClevrCogsciCurriculum_test",
                        action='store_true',
                        help='Runs tests for cogsci curriculum methods in makeClevrTasks.py, which controls loading the CLEVR task datasets.')
                        
                    

def run_unit_tests(args):
    if args.pop("run_cogsci_utils_test"):
        test_cogsci_utils.test_all()
        exit(0)
    
    if args.pop("run_makeClevrCogsciCurriculum_test"):
        test_makeClevrCogsciCurriculum.test_all()
        exit(0)
    

def main(args):
    print("##### running main_cogsci: the CogSci 2020 code. ######")
    # Entrypoint for running any unit tests.
    run_unit_tests(args)
    
    # Load a set of conditions.
    dreamcoder_condition_args = cogsci_utils.load_condition_args(args)
    # Load a set of curricula.
        # Number of data splits to try as an argument.
    curricula, language_dataset = makeClevrTasks.loadAllOrderedCurricula(args)
    
    # Load the primitives and optionally run tests with the primitive set.
    primitive_names = args.pop("primitives")
    primitives = load_clevr_primitives(primitive_names)
    initial_grammar = Grammar.uniform(primitives)
    print("Using starting grammar")
    print(initial_grammar)
    
    # Get the evaluation timeout for each task, and the iterations we should run as a whole.
    evaluation_timeout = args.pop("evaluationTimeout")
    eprint(f"Now running with an evaluation timeout of [{evaluation_timeout}].")
    convert_iterations_to_training_task_epochs(args, train_tasks)

    # Each one of these is a single experiment.
    for condition in dreamcoder_condition_args:
        condition_args = dreamcoder_condition_args[condition]
        for curriculum in curricula:
            curriculum_train, curriculum_test = curricula[curriculum]
            print(f"Launching cogsci experiment: {condition} | {curriculum}")
            # Create a directory 
            top_level_output_dir = args.pop("topLevelOutputDirectory")
            checkpoint_output_prefix = get_timestamped_output_directory_for_checkpoints(top_level_output_dir=top_level_output_dir, domain_name=DEFAULT_COGSCI_PREFIX)
            
            args["languageDataset"] = language_dataset
            condition_args["tasks"] = curriculum_train
            condition_args["testingTasks"] = curriculum_test
            condition_args["grammar"] = initial_grammar
            condition_args["outputPrefix"] = checkpoint_output_prefix
            condition_args['evaluationTimeout'] = evaluation_timeout
            
            # Utility to pop off any additional arguments that are specific to this domain.
            pop_all_domain_specific_args(args_dict=args, iterator_fn=ecIterator)
            generator = ecIterator(**dreamcoder_condition_args,
                                   **args)
            for result in generator:
                pass