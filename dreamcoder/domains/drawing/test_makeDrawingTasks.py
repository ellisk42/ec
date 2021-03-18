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

def test_loadAllTrainTaskSchedules_no_schedule(args):
    mock_args = MockArgs(taskDatasetDir=DEFAULT_COMPOSITIONAL_LOGO_DATASET,
    trainTestSchedule=None)
    mock_args = vars(mock_args)
    train_task_schedule = to_test.loadAllTrainTaskSchedules(mock_args)
    assert len(train_task_schedule) == 1
    for (train_tasks, test_tasks) in train_task_schedule:
        assert len(train_tasks) > 0
        assert len(test_tasks) > 0
        assert type(train_tasks[0]) == Task

def test_loadAllTrainTaskSchedules_generate_logo(args):
    mock_args = MockArgs(taskDatasetDir="logo_unlimited_5",
                        trainTestSchedule=None)
    mock_args = vars(mock_args)
    train_task_schedule = to_test.loadAllTrainTaskSchedules(mock_args)
    # TODO: generate tasks.

def run_test(test_fn, args):
    """Utility function for running tests"""
    print(f"Running {test_fn.__name__}...")
    test_fn(args)
    print("\n")
    
def test_all(args):
    print("Running tests for makeDrawingTasks....")
    run_test(test_buildDefaultDrawingTasksRegistry, args)
    run_test(test_loadAllTrainTaskSchedules_no_schedule, args)
    run_test(test_loadAllTrainTaskSchedules_generate_logo, args)
    print("....all tests passed!\n")