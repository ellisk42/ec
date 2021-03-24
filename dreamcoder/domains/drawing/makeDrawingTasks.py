"""
makeDrawingTasks.py | Author : Catherine Wong
Utility functions for loading drawing task, image, and language data to construct Task objects.

loadTaskAndLanguageDataset is the main entrypoint function.
"""
import os

import dreamcoder.domains.logo.makeLogoTasks as makeLogoTasks 
from types import SimpleNamespace

TASKS_SUBDIR = "tasks"
LANGUAGE_SUBDIR = "language"
SPLITS = ['train', 'test']
LANGUAGE_JSON_FILES = ["language.json", "vocab.json"]

## Default cached tasks.
DEFAULT_COMPOSITIONAL_LOGO_DIR = "data/logo" # Compositional logo tasks from ICML 2020.
DEFAULT_TIAN_DRAWING_DIR = "data/tian" # Lucas Tian tasks from NeurIPS 2020
ALL_DRAWING_DATASET_DIRS = [
    DEFAULT_COMPOSITIONAL_LOGO_DIR,
    DEFAULT_TIAN_DRAWING_DIR
] # All datasets that will be loaded into the registry by default.
 
## Default generatable task datasets.
GENERATE_COMPOSITIONAL_LOGO_TAG = "logo_unlimited"
GENERATE_ORIGINAL_LOGO_TAG = "logo_original"
GENERATED_TASK_DATASETS = [
    GENERATE_COMPOSITIONAL_LOGO_TAG,
    GENERATE_ORIGINAL_LOGO_TAG
] 

def loadAllTaskAndLanguageDatasets(args):
    """
    Loads the task data for the drawing train and task datasets and converts them into train/test schedules of typed DreamCoder Task objects.
    
    If the task dataset is not yet generated, optionally generates and caches the new task dataset and corresponding language data.
    
    Loads a language dataset if applicable, and checks that a language dataset exists.
    
    Returns: and Object with:
        {
            train_test_schedules : [
                ([array of language-annotated Task objects], [array of language-annotated Task objects])
            ],
            language_dataset : [array of string names for the task classes used that will be used to load the corresponding natural language.]
        }
    """
    task_dataset_tag, top_level_data_dir, train_test_schedules = loadAllTrainTaskSchedules(args)
    language_dataset = checkAndLoadAllLanguageDatasets(top_level_data_dir, task_dataset_tag, args)
    return SimpleNamespace(
            train_test_schedules=train_test_schedules,
            language_dataset=language_dataset
    )

def checkAndLoadAllLanguageDatasets(top_level_data_dir, task_dataset_tag, args):
    """
    Checks for the existence of the language dataset dir within the desired dataset. Throws an error if not applicable.
    Assumes the language dataset should be in: {top_level_data_dir}/language/{DATASET}/{LANGUAGE_DATASET_SUBDIR}
    Returns: [top level languageDatasetDir].
    """
    languageDatasetSubdir = args['languageDatasetSubdir']
    if languageDatasetSubdir is None:
        return []
    else:
        full_language_dir = os.path.join(top_level_data_dir, LANGUAGE_SUBDIR, task_dataset_tag, languageDatasetSubdir)
        for split in SPLITS:
            full_language_split_dir = os.path.join(full_language_dir, split)
            for language_file in LANGUAGE_JSON_FILES:
                full_language_file_path = os.path.join(full_language_split_dir, language_file)
                print(full_language_file_path)
                assert os.path.exists(full_language_file_path)
        return [full_language_dir]
            

def getDefaultCachedDrawingDatasets():
    """Convenience wrapper around registry that prints human readable available default drawing tasks, as well as tasks that can be generated."""
    return list(buildDefaultDrawingTasksRegistry().keys())
    
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
    task_dataset = args["taskDatasetDir"]
    generated_dataset = args['generateTaskDataset']
    ## Load a cached task dataset from the registry.
    if task_dataset in drawing_tasks_registry:
        task_dataset_tag = task_dataset
        full_task_directory = drawing_tasks_registry[task_dataset]
        if DEFAULT_COMPOSITIONAL_LOGO_DIR in full_task_directory:
            top_level_data_dir = DEFAULT_COMPOSITIONAL_LOGO_DIR
            full_train, full_test = makeLogoTasks.loadLogoDataset(full_directory=full_task_directory)
        elif DEFAULT_TIAN_DRAWING_DIR in full_task_directory:
            top_level_data_dir = DEFAULT_TIAN_DRAWING_DIR 
            print("Unimplemented: loading the Tian et. al drawing tasks.")
            assert False
        else:
            print(f"Unknown task dataset: {task_dataset}")
            assert False
    ## Generate the tasks from scratch.    
    else:
        assert generated_dataset is not None
        task_dataset_tag, top_level_data_dir, full_train, full_test = generateAndLoadDrawingTaskDataset(args)

    ## Generate the training schedule from the tasks.
    train_test_schedule = generateTrainTestScheduleFromTasks(args, full_train, full_test)
    return task_dataset_tag, top_level_data_dir, train_test_schedule

def generateTrainTestScheduleFromTasks(args, full_train, full_test):
    if args["trainTestSchedule"] is None:
        return [(full_train, full_test)]
    else:
        print("Not yet implemented: generating full train and test schedule.")
        assert False

def generateAndLoadDrawingTaskDataset(args):
    """
    Generates Tasks for a full dataset if not already cached. 
    Stores them in {DEFAULT_TASK_DIR}/tasks/{DATASET_TAG}_{NTASKS}.
    Automatically generates language when available.
    Expects: 
        args.taskDatasets: subdirectory name for the tasks dataset.
    Returns train, test: [Task object], [Task object]
    """
    generated_dataset_class = args["generateTaskDataset"] 
    n_tasks = args["nGeneratedTasks"]
    assert type(n_tasks) == type(0) # Must be an int
    assert generated_dataset_class in GENERATED_TASK_DATASETS
    
    n_tasks_tag = n_tasks if n_tasks > 0 else "all"
    task_dataset_tag = f"{generated_dataset_class}_{n_tasks_tag}"
    
    if generated_dataset_class == GENERATE_COMPOSITIONAL_LOGO_TAG:
        task_dataset_dir = os.path.join(DEFAULT_COMPOSITIONAL_LOGO_DIR, TASKS_SUBDIR)
        top_level_data_dir = DEFAULT_COMPOSITIONAL_LOGO_DIR
        return task_dataset_tag,top_level_data_dir, makeLogoTasks.generateCompositionalLogoDataset(task_dataset_dir=task_dataset_dir,
                                         task_dataset=task_dataset_tag,
                                         n_tasks=n_tasks)
    elif generated_dataset_class == GENERATE_ORIGINAL_LOGO_TAG:
        task_dataset_dir = os.path.join(DEFAULT_COMPOSITIONAL_LOGO_DIR, TASKS_SUBDIR)
        top_level_data_dir = DEFAULT_COMPOSITIONAL_LOGO_DIR
        return task_dataset_tag, top_level_data_dir, makeLogoTasks.generateOriginalLogoDataset(task_dataset_dir=task_dataset_dir,
                                    task_dataset=task_dataset_tag)
    else:
        print(f"Not yet supported: {task_dataset_class}")
        assert False