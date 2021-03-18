"""
makeDrawingTasks.py | Author : Catherine Wong
Utility functions for loading drawing task, image, and language data to construct Task objects.

loadTaskAndLanguageDataset is the main entrypoint function.
"""
import os

from dreamcoder.domains.logo.makeLogoTasks import loadLogoDataset, generateCompositionalLogoDataset

TASKS_SUBDIR = "tasks"
LANGUAGE_SUBDIR = "language"
DEFAULT_COMPOSITIONAL_LOGO_DIR = "data/logo" # Compositional logo tasks from ICML 2020.
DEFAULT_TIAN_DRAWING_DIR = "data/tian" # Lucas Tian tasks from NeurIPS 2020
ALL_DRAWING_DATASET_DIRS = [
    DEFAULT_COMPOSITIONAL_LOGO_DIR,
    DEFAULT_TIAN_DRAWING_DIR
] # All datasets that will be loaded into the registry by default.
 
LOGO_UNLIMITED_TAG = "logo_unlimited"
LOGO_ORIGINAL = "logo_original"
GENERATED_TASK_DIRS = [
    LOGO_UNLIMITED_TAG,
    LOGO_ORIGINAL
] 

def loadAllTaskAndLanguageDatasets(args):
    """
    Loads the task data for the drawing train and task datasets and converts them into train/test schedules of typed DreamCoder Task objects.
    
    If the task dataset is not yet generated, optionally generates and caches the new task dataset and corresponding language data.
    
    If language data is provided, annotates the tasks with their provided language.
    
    Mutates args['languageDatasetDir'] = {taskDatasetDir}/language
    
    Returns: and object with:
        {
            train_test_schedules : [
                ([array of language-annotated Task objects], [array of language-annotated Task objects])
            ],
            language_dataset : [array of string names for the task classes used that will be used to load the corresponding natural language.]
        }
    """
    train_test_schedules = loadAllTrainTaskSchedules(args)
    language_dataset 

def getDefaultDrawingTaskDirectories():
    """Convenience wrapper around registry that prints human readable available default drawing tasks, as well as tasks that can be generated."""
    return list(buildDefaultDrawingTasksRegistry().keys()) + GENERATED_TASK_DIRS
def buildDefaultDrawingTasksRegistry():
    """
    Builds a registry of default available task datasets.
    Expects the dataset to be in a {top_level_dir}/tasks directory.
    Returns:
        {dataset_name : full_directory_path}
    """
    drawing_tasks_registry = {}
    for top_level_dir in ALL_DRAWING_DATASET_DIRS:
        tasks_dir = os.path.join(top_level_dir, TASKS_SUBDIR)
        for dataset_subdir in os.listdir(tasks_dir):
            full_directory_path = os.path.join(tasks_dir, dataset_subdir)
            if os.path.isdir(full_directory_path):
                drawing_tasks_registry[dataset_subdir] = full_directory_path
    return drawing_tasks_registry

def generateAndLoadDrawingTaskDataset(args):
    """
    Generates Tasks for a full dataset if not already cached. 
    Expects: 
        args.taskDatasets: subdirectory name for the tasks dataset.
    Returns train, test: [Task object], [Task object]
    """
    # TODO (cathywong)
    drawing_tasks_registry = buildDefaultDrawingTasksRegistry()
    task_dataset = args["taskDatasetDir"] 
    full_task_directory = drawing_tasks_registry[task_dataset]
    if DEFAULT_COMPOSITIONAL_LOGO_DIR in full_task_directory:
        generateCompositionalLogoDataset(task_dataset, args)
    else:
        print("Dataset for ")
        assert False

def generateTrainTestScheduleFromTasks(args, full_train, full_test):
    assert False
    
def loadAllTrainTaskSchedules(args):
    """
    Loads all of the training and testing task schedules for the drawing datasets, and converts them into DreamCoder Task objects.
    
    Expects:
        args.taskDatasets: subdirectory name for the tasks dataset.
        args.trainTestScheduleFile: if None, returns a single training/testing schedule.
    Returns:
        [
            ([Task object], [Task object])
        ]
    """
    drawing_tasks_registry = buildDefaultDrawingTasksRegistry()
    ## Load a cached task dataset from the registry.
    if args["taskDatasetDir"] in drawing_tasks_registry:
        full_task_directory = drawing_tasks_registry[args["taskDatasetDir"]]
        if DEFAULT_COMPOSITIONAL_LOGO_DIR in full_task_directory:
            full_train, full_test = loadLogoDataset(full_directory=full_task_directory)
        elif DEFAULT_TIAN_DRAWING_DIR in full_task_directory:
            # Load the Tian dataset.
            assert False
        else:
            print("Unknown task dataset.")
            assert False
            
    else:
    ## Generate the tasks from scratch.
        full_train, full_test = generateAndLoadDrawingTaskDataset(args)
    
    ## Generate the training schedule from the tasks.
    if args["trainTestSchedule"] is None:
        return [(full_train, full_test)]
    else:
        return generateTrainTestScheduleFromTasks(args, full_train, full_test)