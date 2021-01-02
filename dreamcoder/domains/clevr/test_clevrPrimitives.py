import dreamcoder.domains.clevr.clevrPrimitives as to_test
import dreamcoder.domains.clevr.makeClevrTasks as makeClevrTasks
from types import SimpleNamespace as MockArgs
"""
Tests for clevrPrimitives.py | Author: Catherine Wong
Tests for the CLEVR Python implementations of primitives.

All tests are manually added to a 'test_all' function.
"""
DEFAULT_CLEVR_DATASET_DIR = 'data/clevr'
DATASET_CACHE = dict()

def get_train_task_datasets(task_dataset):
    mock_args = MockArgs(taskDatasetDir=DEFAULT_CLEVR_DATASET_DIR,
    curriculumDatasets=[],taskDatasets=[task_dataset])
    if task_dataset not in DATASET_CACHE:
        all_train_tasks, all_test_tasks = makeClevrTasks.loadAllTaskDatasets(mock_args)
        DATASET_CACHE[task_dataset] = all_train_tasks
    else:
        all_train_tasks = DATASET_CACHE[task_dataset]
    return all_train_tasks
    
def get_default_localization_task():
    task_dataset = get_train_task_datasets(task_dataset="2_localization")
    default_task = task_dataset[0]
    print(f"Testing task: {default_task.name}")
    return default_task

def get_default_localization_task_multiple_filter():
    task_dataset = get_train_task_datasets(task_dataset="2_localization")
    default_task = task_dataset[55]
    print(f"Testing task: {default_task.name}")
    return default_task

def get_default_zero_hop_task_count():
    """How many metal things are there?"""
    task_dataset = get_train_task_datasets(task_dataset="1_zero_hop")
    default_task = task_dataset[0]
    print(f"Testing task: {default_task.name}")
    return default_task
    
def get_default_zero_hop_task_query_shape():
    """What is the shape of the small yellow thing?"""
    task_dataset = get_train_task_datasets(task_dataset="1_zero_hop")
    default_task = task_dataset[9]
    print(f"Testing task: {default_task.name}")
    return default_task

def get_default_zero_hop_task_query_material():
    """What is the purple thing made of?"""
    task_dataset = get_train_task_datasets(task_dataset="1_zero_hop")
    default_task = task_dataset[12]
    print(f"Testing task: {default_task.name}")
    return default_task

def get_default_zero_hop_task_query_color():
    """What color is the metal sphere?"""
    task_dataset = get_train_task_datasets(task_dataset="1_zero_hop")
    default_task = task_dataset[19]
    print(f"Testing task: {default_task.name}")
    return default_task

def get_default_zero_hop_task_query_size():
    """The brown metal thing has what size?"""
    task_dataset = get_train_task_datasets(task_dataset="1_zero_hop")
    default_task = task_dataset[28]
    print(f"Testing task: {default_task.name}")
    return default_task

def get_default_one_hop_count():
    """How many things are right the large cylinder?"""
    task_dataset = get_train_task_datasets(task_dataset="1_one_hop")
    default_task = task_dataset[0]
    print(f"Testing task: {default_task.name}")
    return default_task

def get_default_one_hop_query():
    """There is a thing front the brown thing; how big is it?"""
    task_dataset = get_train_task_datasets(task_dataset="1_one_hop")
    default_task = task_dataset[7]
    print(f"Testing task: {default_task.name}")
    return default_task

def get_default_same_relate_count():
    """How many other things are there of the same size as the cyan thing?"""
    task_dataset = get_train_task_datasets(task_dataset="1_same_relate_restricted")
    default_task = task_dataset[0]
    print(f"Testing task: {default_task.name}")
    return default_task

def get_default_same_relate_query():
    """There is a red thing that is the same size as the metal cylinder; what shape is it?"""
    task_dataset = get_train_task_datasets(task_dataset="1_same_relate_restricted")
    default_task = task_dataset[14]
    print(f"Testing task: {default_task.name}")
    return default_task

def get_default_compare_integer_less_than():
    """Is the number of cyan rubber things less than the number of large cylinders?"""
    task_dataset = get_train_task_datasets(task_dataset="1_compare_integer")
    default_task = task_dataset[2]
    print(f"Testing task: {default_task.name}")
    return default_task

def get_default_compare_integer_greater_than():
    """Is the number of large cylinders greater than the number of small rubber spheres?"""
    task_dataset = get_train_task_datasets(task_dataset="1_compare_integer")
    default_task = task_dataset[3]
    print(f"Testing task: {default_task.name}")
    return default_task

def get_default_single_or():
    """How many cylinders are brown things or small rubber things?"""
    task_dataset = get_train_task_datasets(task_dataset="1_single_or")
    default_task = task_dataset[22]
    print(f"Testing task: {default_task.name}")
    return default_task

def get_default_remove():
    """What if you removed all of the blue metal things?"""
    task_dataset = get_train_task_datasets(task_dataset="1_remove")
    default_task = task_dataset[0]
    print(f"Testing task: {default_task.name}")
    return default_task

def get_default_remove_query():
    """If you removed the red things, how many spheres would be left?"""
    task_dataset = get_train_task_datasets(task_dataset="1_remove")
    default_task = task_dataset[11]
    print(f"Testing task: {default_task.name}")
    return default_task

def get_default_transform():
    """What if the gray sphere became a small green metal sphere?"""
    task_dataset = get_train_task_datasets(task_dataset="1_transform")
    default_task = task_dataset[0]
    print(f"Testing task: {default_task.name}")
    return default_task

def get_default_transform_query():
    """If all of the blue spheres became brown, how many brown things would there be?"""
    task_dataset = get_train_task_datasets(task_dataset="1_transform")
    default_task = task_dataset[7]
    print(f"Testing task: {default_task.name}")
    return default_task
    
# Tests that we can solve each of the individual question classes with the reimplementation.
def test_localization_task_original_primitives():
    print(f"Running test_localization_task_original_primitives")
    localization_task = get_default_localization_task()

def test_localization_task_multiple_filter_original_primitives():
    print(f"Running test_localization_task_original_primitives")
    localization_task = get_default_localization_task_multiple_filter()

# Tests that we can solve each of the individual question classes with the LISP-based primitive set.

    

# Test enumeration.


def test_all():
    print("Running tests for clevrPrimitives....")
    
    test_localization_task_original_primitives()
    test_localization_task_multiple_filter_original_primitives()
    pass