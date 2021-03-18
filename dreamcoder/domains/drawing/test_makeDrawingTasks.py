import dreamcoder.domains.drawing.makeDrawingTasks as to_test
from dreamcoder.task import Task
"""
Tests for makeClevrTasks.py | Author: Catherine Wong
Tests for making and loading drawing tasks.

All tests are manually added to a 'test_all' function.
"""
from types import SimpleNamespace as MockArgs
import os

DEFAULT_COMPOSITIONAL_LOGO_DATASET = "logo_unlimited_200"

def test_buildDefaultDrawingTasksRegistry(args):
    drawing_tasks_registry = to_test.buildDefaultDrawingTasksRegistry()
    assert len(drawing_tasks_registry) > 0
    for (dataset_name, directory_path) in drawing_tasks_registry.items():
        # Is at least one of the directories in the title?
        contains_default_top_level_directory = False
        for top_level_dir in to_test.ALL_DRAWING_DATASET_DIRS:
            if top_level_dir in directory_path:
                contains_default_top_level_directory = True
        
        assert contains_default_top_level_directory
        # Is it a real directory?
        assert os.path.isdir(directory_path)

def test_loadAllTaskAndLanguageDatasets_no_language(args):
    mock_args = MockArgs(taskDatasetDir=DEFAULT_COMPOSITIONAL_LOGO_DATASET,
    languageDatasetSubdir=None,
    trainTestSchedule=None,
    generateTaskDataset=None)
    mock_args = vars(mock_args)
    
    task_and_language_object = to_test.loadAllTaskAndLanguageDatasets(mock_args)
    
    train_task_schedule = task_and_language_object.train_test_schedules
    assert len(train_task_schedule) == 1
    for (train_tasks, test_tasks) in train_task_schedule:
        assert len(train_tasks) > 0
        assert len(test_tasks) > 0
        assert type(train_tasks[0]) == Task
    assert len(task_and_language_object.language_dataset) == 0

def test_loadAllTaskAndLanguageDatasets_synth_language(args):
    mock_args = MockArgs(taskDatasetDir=DEFAULT_COMPOSITIONAL_LOGO_DATASET,
    languageDatasetSubdir="synthetic",
    trainTestSchedule=None,
    generateTaskDataset=None)

def test_loadAllTaskAndLanguageDatasets_human_language(args):
    mock_args = MockArgs(taskDatasetDir=DEFAULT_COMPOSITIONAL_LOGO_DATASET,
    languageDatasetSubdir="human",
    trainTestSchedule=None,
    generateTaskDataset=None)
    
def test_loadAllTrainTaskSchedules_no_schedule(args):
    mock_args = MockArgs(taskDatasetDir=DEFAULT_COMPOSITIONAL_LOGO_DATASET,
    trainTestSchedule=None,
    generateTaskDataset=None)
    mock_args = vars(mock_args)
    task_dataset_tag, train_task_schedule = to_test.loadAllTrainTaskSchedules(mock_args)
    assert len(train_task_schedule) == 1
    for (train_tasks, test_tasks) in train_task_schedule:
        assert len(train_tasks) > 0
        assert len(test_tasks) > 0
        assert type(train_tasks[0]) == Task
    assert task_dataset_tag  == DEFAULT_COMPOSITIONAL_LOGO_DATASET

def test_loadAllTrainTaskSchedules_generate_logo_compositional(args):
    nGeneratedTasks = 5
    mock_args = MockArgs(generateTaskDataset=to_test.GENERATE_COMPOSITIONAL_LOGO_TAG,
            nGeneratedTasks=nGeneratedTasks,
            trainTestSchedule=None,
            taskDatasetDir=None)
    mock_args = vars(mock_args)
    task_dataset_tag, train_task_schedule = to_test.loadAllTrainTaskSchedules(mock_args)
    assert len(train_task_schedule) == 1
    for (train_tasks, test_tasks) in train_task_schedule:
        assert len(train_tasks) > 0
        assert len(test_tasks) > 0
        assert type(train_tasks[0]) == Task
    assert task_dataset_tag == f"{to_test.GENERATE_COMPOSITIONAL_LOGO_TAG}_{nGeneratedTasks}"

def test_loadAllTrainTaskSchedules_generate_logo_original(args):
    mock_args = MockArgs(generateTaskDataset=to_test.GENERATE_ORIGINAL_LOGO_TAG,
            nGeneratedTasks=-1,
            trainTestSchedule=None,
            taskDatasetDir=None)
    mock_args = vars(mock_args)
    task_dataset_tag, train_task_schedule = to_test.loadAllTrainTaskSchedules(mock_args)
    assert len(train_task_schedule) == 1
    for (train_tasks, test_tasks) in train_task_schedule:
        assert len(train_tasks) > 0
        assert len(test_tasks) > 0
        assert type(train_tasks[0]) == Task
    assert task_dataset_tag == f"{to_test.GENERATE_ORIGINAL_LOGO_TAG}_all"

def run_test(test_fn, args):
    """Utility function for running tests"""
    print(f"Running {test_fn.__name__}...")
    test_fn(args)
    print("\n")
    
def test_all(args):
    print("Running tests for makeDrawingTasks....")
    run_test(test_buildDefaultDrawingTasksRegistry, args)
    run_test(test_loadAllTaskAndLanguageDatasets_no_language, args)
    # run_test(test_loadAllTrainTaskSchedules_no_schedule, args)
    # run_test(test_loadAllTrainTaskSchedules_generate_logo_compositional, args)
    # run_test(test_loadAllTrainTaskSchedules_generate_logo_original, args)
    print("....all tests passed!\n")