from dreamcoder.utilities import *
from dreamcoder.dreamcoder import ecIterator, default_wake_generative

import dreamcoder.domains.drawing.makeDrawingTasks as makeDrawingTasks
import dreamcoder.domains.drawing.test_makeDrawingTasks as test_makeDrawingTasks

import dreamcoder.domains.drawing.drawingPrimitives
import dreamcoder.domains.drawing.test_drawingPrimitives as test_drawingPrimitives
"""
main.py (drawing)  | Author: Catherine Wong.
This is the main file for compositional graphics tasks that require generating programs which draw images. (It is usually launched via the bin/drawing.py convenience file.)

This dataset was introduced for a series of cognitive and computational experiments for language and drawing related tasks.
It contains convenience arguments for using both the LOGO graphics-related domain (also in logo.py) and the human experiment stimuli in Tian et. al 2020.

Example usage: 
    python bin/drawing.py 
        --taskDatasetDir logo_unlimited_200
        --languageDatasetDir synthetic
        --primitives logo

Example tests: TODO
"""

DEFAULT_DRAWING_DOMAIN_NAME_PREFIX = "drawing"
DOMAIN_SPECIFIC_ARGS = {
    "grammar" : None,
    "tasks" : None, # Training tasks.
    "testingTasks" : None,
    "outputPrefix": None, # Output prefix for the checkpoint files,
}

# Domain specific command line arguments.
def drawing_options(parser):
    ## Dataset loading options.
    parser.add_argument("--taskDatasetDir", type=str,
                        choices=makeDrawingTasks.getDefaultCachedDrawingDatasets(),
                        help="Sub directory name for the task dataset. Recovers the top-level tasks dataset dir based on the unique subdirectory. Must be a cached dataset.")
    parser.add_argument("--languageDatasetSubdir",
                        help="Language dataset subdirectory. Expects the subdirectory to exist within the task dataset subdirectory, e.g. {taskDatasetDir}/language/{languageSubdir}")
    parser.add_argument("--trainTestSchedule", type=str,
                        help="[Currently unimplemented] Optional file for building subschedules of train and testing stimuli -- for instance, if replicating the behavior of multiple subjects on the same smaller schedule of tasks. If not included, generates a single schedule of the full dataset.")
    parser.add_argument("--topLevelOutputDirectory",
                        default=DEFAULT_OUTPUT_DIRECTORY, # Defined in utilities.py
                        help="Top level directory in which to store outputs. By default, this is the experimentOutputs directory.")
    parser.add_argument("--generateTaskDataset", type=str,
                        choices=makeDrawingTasks.GENERATED_TASK_DATASETS,
                        help="If provided, generates a task dataset from scratch. Must specify nGeneratedTasks.")
    parser.add_argument("--nGeneratedTasks",
                        type=int,
                        help="If {taskDatasetDir} is not a cached directory, generates n tasks or {-1} to generate all takss from that generator.")
    
    # Experiment iteration parameters.
    parser.add_argument("--primitives",
                        nargs="*",
                        help="Which primitives to use. Choose from: [logo, tian_{library_version}].")
                        
    
    # Test functionalities.
    parser.add_argument("--run_makeDrawingTasks_test",
                        action='store_true',
                        help='Runs tests for makeDrawingTasks.py, which controls loading the drawing task datasets.')
    parser.add_argument("--run_drawingPrimitives_test",
                        action='store_true',
                        help='Runs tests for drawingPrimitives.py, which controls initial primitives for the drawing domain.')

def run_unit_tests(args):
    # Runs all unit tests for this domain.
    if args.pop("run_makeDrawingTasks_test"):
        test_makeDrawingTasks.test_all(args)
        exit(0)
    if args.pop("run_drawingPrimitives_test"):
        test_drawingPrimitives.test_all(args)
        exit(0)

def main(args):
    # Entrypoint for running any unit tests.
    run_unit_tests(args)
    
    # Load the train and testing schedule. 
    # Generates and caches the dataset if an uncached taskDatasetDir is provided.
    task_and_language_schedule = makeDrawingTasks.loadAllTaskAndLanguageDatasets(args)
    train_test_schedules = task_and_language_schedule.train_test_schedules
    language_dataset = task_and_language_schedule.language_dataset
    
    
    # Load the initial grammar.
    initial_grammar = drawingPrimitives.load_initial_grammar(args)
    
    # Create the output directory for the experiment.
    top_level_output_dir = args.pop("topLevelOutputDirectory")
    checkpoint_output_prefix = get_timestamped_output_directory_for_checkpoints(top_level_output_dir=top_level_output_dir, domain_name=DEFAULT_DRAWING_DOMAIN_NAME_PREFIX)
    
    # Set all of the domain specific arguments.
    
    # Run the integration test immediately before we remove all domain-specific arguments.
    
    # Utility to pop off any additional arguments that are specific to this domain.
