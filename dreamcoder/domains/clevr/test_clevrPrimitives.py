from dreamcoder.program import Primitive, Program

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

def check_task_evaluation(test_task, raw_program, should_succeed=True):
    to_test.clevr_original_v1_primitives()
    print(f"Testing program: {raw_program}")
    p = Program.parse(raw_program)
    test_pass = test_task.check(p, timeout=1000)
    print(f"{test_task.name} | pass: {test_pass}")
    assert test_pass == should_succeed

def get_train_task_datasets(task_dataset):
    mock_args = vars(MockArgs(taskDatasetDir=DEFAULT_CLEVR_DATASET_DIR,
    curriculumDatasets=[],taskDatasets=[task_dataset]))
    if task_dataset not in DATASET_CACHE:
        all_train_tasks, all_test_tasks = makeClevrTasks.loadAllTaskDatasets(mock_args)
        DATASET_CACHE[task_dataset] = all_train_tasks
    else:
        all_train_tasks = DATASET_CACHE[task_dataset]
    return all_train_tasks

def get_default_localization_task():
    """Find the large things."""
    task_dataset = get_train_task_datasets(task_dataset="2_localization")
    default_task = task_dataset[0]
    print(f"Testing task: {default_task.name}")
    return default_task

def get_default_localization_task_multiple_filter():
    """Find the small cube."""
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
    task_dataset = get_train_task_datasets(task_dataset="2_remove")
    default_task = task_dataset[0]
    print(f"Testing task: {default_task.name}")
    return default_task

def get_default_remove_query():
    """If you removed the red things, how many spheres would be left?"""
    task_dataset = get_train_task_datasets(task_dataset="2_remove")
    default_task = task_dataset[11]
    print(f"Testing task: {default_task.name}")
    return default_task

def get_default_transform():
    """What if the gray sphere became a small green metal sphere?"""
    task_dataset = get_train_task_datasets(task_dataset="2_transform")
    default_task = task_dataset[0]
    print(f"Testing task: {default_task.name}")
    return default_task

def get_default_transform_query():
    """If all of the blue spheres became brown, how many brown things would there be?"""
    task_dataset = get_train_task_datasets(task_dataset="2_transform")
    default_task = task_dataset[7]
    print(f"Testing task: {default_task.name}")
    return default_task

# Tests error handling on common error scenarios.
def test_relate_not_in_list():
    # Tests if we can fail safely if the query object is not in the object list.
    """How many things are right the large cylinder?"""
    one_task = get_default_one_hop_count()
    # Bootstrapped primitives.
    is_large = "(clevr_eq_size clevr_large (clevr_query_size $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_large} (clevr_add $1 $0) $0)))"
    filter_large = f"(clevr_fold $0 clevr_empty {fold_fn})"
    
    is_cylinder = "(clevr_eq_shape clevr_cylinder (clevr_query_shape $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_cylinder} (clevr_add $1 $0) $0)))"
    filter_large_cylinder = f"(clevr_fold {filter_large} clevr_empty {fold_fn})"
    get_single_object = f"(clevr_car {filter_large_cylinder})"
    empty_list = f"clevr_empty"
    get_right_of = f"(clevr_relate {get_single_object} clevr_right {empty_list})"
    raw_program_bootstrap = f"(lambda (clevr_count {get_right_of}))"
    check_task_evaluation(one_task, raw_program_bootstrap, should_succeed=False)
    print("\n")
    return one_task, raw_program_bootstrap
    
def test_relate_no_relations():
    # Tests that we can fail safely if there is no object with any relations in the list.
    one_task = get_default_one_hop_count()
    # Bootstrapped primitives.
    is_large = "(clevr_eq_size clevr_large (clevr_query_size $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_large} (clevr_add $1 $0) $0)))"
    filter_large = f"(clevr_fold $0 clevr_empty {fold_fn})"
    
    is_cylinder = "(clevr_eq_shape clevr_cylinder (clevr_query_shape $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_cylinder} (clevr_add $1 $0) $0)))"
    filter_large_cylinder = f"(clevr_fold {filter_large} clevr_empty {fold_fn})"
    get_single_object = f"(clevr_car {filter_large_cylinder})"

    get_right_of = f"(clevr_relate {get_single_object} clevr_right {filter_large_cylinder})"
    raw_program_bootstrap = f"(lambda (clevr_count {get_right_of}))"
    check_task_evaluation(one_task, raw_program_bootstrap, should_succeed=False)
    print("\n")
    return one_task, raw_program_bootstrap

def test_union_same_lists():
    # Tests that we can handle unioning the same list twice..
    """How many cylinders are brown things or small rubber things?"""
    single_or_task = get_default_single_or()

    # Bootstrap primitives
    is_small = "(clevr_eq_size clevr_small (clevr_query_size $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_small} (clevr_add $1 $0) $0)))"
    filter_small = f"(clevr_fold $0 clevr_empty {fold_fn})"
    
    is_rubber = "(clevr_eq_material clevr_rubber (clevr_query_material $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_rubber} (clevr_add $1 $0) $0)))"
    filter_small_rubber = f"(clevr_fold {filter_small} clevr_empty {fold_fn})"
    
    is_brown = "(clevr_eq_color clevr_brown (clevr_query_color $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_brown} (clevr_add $1 $0) $0)))"
    filter_brown = f"(clevr_fold $0 clevr_empty {fold_fn})"
    union = f"(clevr_union {filter_brown} {filter_brown})"
    union_again = f"(clevr_union {union} {filter_small_rubber})"
    
    is_cylinder = "(clevr_eq_shape clevr_cylinder (clevr_query_shape $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_cylinder} (clevr_add $1 $0) $0)))"
    filter_cylinder = f"(clevr_fold {union_again} clevr_empty {fold_fn})"
    raw_program_bootstrap = f"(lambda (clevr_count {filter_cylinder}))"
    check_task_evaluation(single_or_task, raw_program_bootstrap)
    print("\n")
    return single_or_task, raw_program_bootstrap

def test_intersect_no_intersection():
    # Tests that we can handle intersections with no intersection.
    single_or_task = get_default_single_or()

    # Bootstrap primitives
    is_small = "(clevr_eq_size clevr_small (clevr_query_size $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_small} (clevr_add $1 $0) $0)))"
    filter_small = f"(clevr_fold $0 clevr_empty {fold_fn})"
    
    is_rubber = "(clevr_eq_material clevr_rubber (clevr_query_material $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_rubber} (clevr_add $1 $0) $0)))"
    filter_small_rubber = f"(clevr_fold {filter_small} clevr_empty {fold_fn})"
    
    is_brown = "(clevr_eq_color clevr_brown (clevr_query_color $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_brown} (clevr_add $1 $0) $0)))"
    filter_brown = f"(clevr_fold $0 clevr_empty {fold_fn})"
    needless_intersection = f"(clevr_intersect clevr_empty clevr_empty)"
    union = f"(clevr_union {filter_brown} {needless_intersection})"
    union_again = f"(clevr_union {union} {filter_small_rubber})"
    
    is_cylinder = "(clevr_eq_shape clevr_cylinder (clevr_query_shape $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_cylinder} (clevr_add $1 $0) $0)))"
    filter_cylinder = f"(clevr_fold {union_again} clevr_empty {fold_fn})"
    raw_program_bootstrap = f"(lambda (clevr_count {filter_cylinder}))"
    check_task_evaluation(single_or_task, raw_program_bootstrap)
    print("\n")
    return single_or_task, raw_program_bootstrap

def test_difference_empty_lists():
    # Tests that we can handle differences between empty lists.
    single_or_task = get_default_single_or()

    # Bootstrap primitives
    is_small = "(clevr_eq_size clevr_small (clevr_query_size $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_small} (clevr_add $1 $0) $0)))"
    filter_small = f"(clevr_fold $0 clevr_empty {fold_fn})"
    
    is_rubber = "(clevr_eq_material clevr_rubber (clevr_query_material $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_rubber} (clevr_add $1 $0) $0)))"
    filter_small_rubber = f"(clevr_fold {filter_small} clevr_empty {fold_fn})"
    
    is_brown = "(clevr_eq_color clevr_brown (clevr_query_color $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_brown} (clevr_add $1 $0) $0)))"
    filter_brown = f"(clevr_fold $0 clevr_empty {fold_fn})"
    needless_difference = f"(clevr_difference clevr_empty clevr_empty)"
    union = f"(clevr_union {filter_brown} {needless_difference})"
    union_again = f"(clevr_union {union} {filter_small_rubber})"
    
    is_cylinder = "(clevr_eq_shape clevr_cylinder (clevr_query_shape $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_cylinder} (clevr_add $1 $0) $0)))"
    filter_cylinder = f"(clevr_fold {union_again} clevr_empty {fold_fn})"
    raw_program_bootstrap = f"(lambda (clevr_count {filter_cylinder}))"
    check_task_evaluation(single_or_task, raw_program_bootstrap)
    print("\n")
    return single_or_task, raw_program_bootstrap

def test_difference_one_empty_list():
    # Tests that we can handle differences between one empty list.
    single_or_task = get_default_single_or()

    # Bootstrap primitives
    is_small = "(clevr_eq_size clevr_small (clevr_query_size $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_small} (clevr_add $1 $0) $0)))"
    filter_small = f"(clevr_fold $0 clevr_empty {fold_fn})"
    
    is_rubber = "(clevr_eq_material clevr_rubber (clevr_query_material $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_rubber} (clevr_add $1 $0) $0)))"
    filter_small_rubber = f"(clevr_fold {filter_small} clevr_empty {fold_fn})"
    
    is_brown = "(clevr_eq_color clevr_brown (clevr_query_color $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_brown} (clevr_add $1 $0) $0)))"
    filter_brown = f"(clevr_fold $0 clevr_empty {fold_fn})"
    needless_difference = f"(clevr_difference clevr_empty {filter_brown})"
    union = f"(clevr_union {filter_brown} {needless_difference})"
    union_again = f"(clevr_union {union} {filter_small_rubber})"
    
    is_cylinder = "(clevr_eq_shape clevr_cylinder (clevr_query_shape $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_cylinder} (clevr_add $1 $0) $0)))"
    filter_cylinder = f"(clevr_fold {union_again} clevr_empty {fold_fn})"
    raw_program_bootstrap = f"(lambda (clevr_count {filter_cylinder}))"
    check_task_evaluation(single_or_task, raw_program_bootstrap)
    print("\n")
    return single_or_task, raw_program_bootstrap

def test_car_empty_list():
    # Tests that we can fail gracefully on taking the car of an empty list.
    zero_task = get_default_zero_hop_task_query_shape()
    # Bootstrapped primitives.
    get_single_object = f"(clevr_car clevr_empty)"
    raw_program_bootstrap = f"(lambda (clevr_query_shape {get_single_object}))"
    check_task_evaluation(zero_task, raw_program_bootstrap, should_succeed=False)
    print("\n")
    return zero_task, raw_program_bootstrap

def test_add_duplicate():
    # Tests that we can incorrectly try adding another object to a list that already has it.
    """There is a thing front the brown thing; how big is it?"""
    one_task = get_default_one_hop_query()
    # Bootstrapped primitives.
    is_brown = "(clevr_eq_color clevr_brown (clevr_query_color $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_brown} (clevr_add $1 $0) $0)))"
    filter_brown = f"(clevr_fold $0 clevr_empty {fold_fn})"
    get_single_object = f"(clevr_car {filter_brown})"
    get_front_of = f"(clevr_relate {get_single_object} clevr_front $0)"
    get_single_object = f"(clevr_car {get_front_of})"
    add_to_empty = f"(clevr_add {get_single_object} clevr_empty)"
    add_again = f"(clevr_add {get_single_object} {add_to_empty})"
    get_single_object = f"(clevr_car {add_again})"
    raw_program_bootstrap = f"(lambda (clevr_query_size {get_single_object}))"
    check_task_evaluation(one_task, raw_program_bootstrap)
    print("\n")
    return one_task, raw_program_bootstrap

def test_if_malformed():
    # Tests that we can fail gracefully if the IF statement doesn't work.
    # """There is a thing front the brown thing; how big is it?"""
    one_task = get_default_one_hop_query()
    # Bootstrapped primitives.
    empty_object = f"(clevr_car clevr_empty)"
    is_brown = f"(clevr_eq_color clevr_brown (clevr_query_color {empty_object}))"
    fold_fn = f"(lambda (lambda (clevr_if {is_brown} (clevr_add $1 $0) $0)))"
    filter_brown = f"(clevr_fold $0 clevr_empty {fold_fn})"
    get_single_object = f"(clevr_car {filter_brown})"
    get_front_of = f"(clevr_relate {get_single_object} clevr_front $0)"
    get_single_object = f"(clevr_car {get_front_of})"
    add_to_empty = f"(clevr_add {get_single_object} clevr_empty)"
    add_again = f"(clevr_add {get_single_object} {add_to_empty})"
    get_single_object = f"(clevr_car {add_again})"
    raw_program_bootstrap = f"(lambda (clevr_query_size {get_single_object}))"
    check_task_evaluation(one_task, raw_program_bootstrap, should_succeed=False)
    print("\n")
    return one_task, raw_program_bootstrap

def test_map_malformed():
    # Tests that we can fail gracefully if the MAP statement doesn't work.
    transform_task = get_default_transform_query()
    empty_object = f"(clevr_car clevr_empty)"
    map_transform = f"(clevr_map (clevr_transform_color clevr_brown) {empty_object})"
    union = f"(clevr_union {map_transform} $0)"
    
    is_brown = "(clevr_eq_color clevr_brown (clevr_query_color $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_brown} (clevr_add $1 $0) $0)))"
    filter_brown = f"(clevr_fold {union} clevr_empty {fold_fn})"
    raw_program_bootstrap = f"(lambda (clevr_count {filter_brown}))"
    check_task_evaluation(transform_task, raw_program_bootstrap, should_succeed=False)
    print("\n")
    return transform_task, raw_program_bootstrap

def test_map_malformed_transform_only():
    # Tests that we can fail gracefully if the MAP statement doesn't work.
    # Bootstrap primitives
    transform_task = get_default_transform_query()
    empty_object = f"(clevr_car clevr_empty)"
    is_brown = f"(clevr_eq_color clevr_brown (clevr_query_color {empty_object}))"
    return_empty = f"(clevr_if {is_brown} {empty_object} {empty_object})"
    
    map_transform_green = f"(clevr_map {return_empty} $0)"
    map_transform_green_metal = f"(clevr_map (clevr_transform_material clevr_metal) {map_transform_green})"
    map_transform_small_green_metal = f"(clevr_map (clevr_transform_size clevr_small) {map_transform_green_metal})"
    raw_program_bootstrap = f"(lambda (clevr_union {map_transform_small_green_metal} $0))"
    check_task_evaluation(transform_task, raw_program_bootstrap, should_succeed=False)
    print("\n")
    return transform_task, raw_program_bootstrap

def test_fold_malformed():
    # Tests that we can fail gracefully if the FOLD statement doesn't work.
    localization_task = get_default_localization_task_multiple_filter()
    # Original primitives.
    empty_object = f"(clevr_car clevr_empty)"
    add_empty = f"(clevr_add {empty_object} clevr_empty)"
    # Bootstrapped primitives.
    is_small = "(clevr_eq_size clevr_small (clevr_query_size $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_small} (clevr_add $1 $0) $0)))"
    filter_small = f"(clevr_fold $0 clevr_empty {fold_fn})"
    
    is_cube = "(clevr_eq_shape clevr_cube (clevr_query_shape $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_cube} (clevr_add $1 $0) {add_empty})))"
    raw_program_bootstrap = f"(lambda (clevr_fold {filter_small} clevr_empty {fold_fn}))"
    check_task_evaluation(localization_task, raw_program_bootstrap, should_succeed=False)
    print("\n")
    return localization_task, raw_program_bootstrap

    
# Tests that we can solve each of the individual question classes with the reimplementation.
def test_localization_task_original_primitives():
    """Find the large things."""
    print(f"Running test_localization_task_original_primitives")
    localization_task = get_default_localization_task()
    
    # Original primitives.
    raw_program_original = "(lambda (clevr_filter_size $0 clevr_large))"
    check_task_evaluation(localization_task, raw_program_original)
    print("\n")
    # Bootstrapped primitives.
    condition = "(clevr_eq_size clevr_large (clevr_query_size $1))"
    fold_fn = f"(lambda (lambda (clevr_if {condition} (clevr_add $1 $0) $0)))"
    raw_program_bootstrap = f"(lambda (clevr_fold $0 clevr_empty {fold_fn}))"
    check_task_evaluation(localization_task, raw_program_bootstrap)
    print("\n")
    return localization_task, raw_program_original, raw_program_bootstrap 

def test_localization_task_original_primitives_base_filter():
    print(f"Running test_localization_task_original_primitives")
    localization_task = get_default_localization_task()
    
    is_large = "(lambda (clevr_eq_size clevr_large (clevr_query_size $0)))"
    raw_program = f"(lambda (clevr_filter {is_large} $0))"
    check_task_evaluation(localization_task, raw_program)
    print("\n")
    return localization_task, raw_program

def test_localization_task_multiple_filter_original_primitives():
    """Find the small cube."""
    print(f"Running test_localization_task_original_primitives")
    localization_task = get_default_localization_task_multiple_filter()
    # Original primitives.
    filter_small = "(clevr_filter_size $0 clevr_small)"
    raw_program_original = f"(lambda (clevr_filter_shape {filter_small} clevr_cube))"
    check_task_evaluation(localization_task, raw_program_original)
    print("\n")
    # Bootstrapped primitives.
    is_small = "(clevr_eq_size clevr_small (clevr_query_size $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_small} (clevr_add $1 $0) $0)))"
    filter_small = f"(clevr_fold $0 clevr_empty {fold_fn})"
    
    is_cube= "(clevr_eq_shape clevr_cube (clevr_query_shape $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_cube} (clevr_add $1 $0) $0)))"
    raw_program_bootstrap = f"(lambda (clevr_fold {filter_small} clevr_empty {fold_fn}))"
    check_task_evaluation(localization_task, raw_program_bootstrap)
    print("\n")
    return localization_task, raw_program_original, raw_program_bootstrap
    
    
def test_zero_hop_task_count_original_primitives():
    print(f"Running test_zero_hop_task_count")
    zero_task = get_default_zero_hop_task_count()
    # Original primitives.
    filter_metal = "(clevr_filter_material $0 clevr_metal)"
    raw_program_original = f"(lambda (clevr_count {filter_metal}))"
    check_task_evaluation(zero_task, raw_program_original)
    print("\n")
    # Bootstrapped primitives.
    is_metal = "(clevr_eq_material clevr_metal (clevr_query_material $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_metal} (clevr_add $1 $0) $0)))"
    filter_metal = f"(clevr_fold $0 clevr_empty {fold_fn})"
    raw_program_bootstrap = f"(lambda (clevr_count {filter_metal}))"
    check_task_evaluation(zero_task, raw_program_bootstrap)
    print("\n")
    return zero_task, raw_program_original, raw_program_bootstrap

def test_zero_hop_task_query_shape_original_primitives():
    """What is the shape of the small yellow thing?"""
    print(f"Running test_zero_hop_task_query_shape")
    zero_task = get_default_zero_hop_task_query_shape()
    # Original primitives.
    filter_yellow = "(clevr_filter_color $0 clevr_yellow)"
    filter_small_yellow = f"(clevr_filter_size {filter_yellow} clevr_small)"
    get_single_object = f"(clevr_unique {filter_small_yellow})"
    raw_program_original = f"(lambda (clevr_query_shape {get_single_object}))"
    check_task_evaluation(zero_task, raw_program_original)
    print("\n")
    # Bootstrapped primitives.
    is_small = "(clevr_eq_size clevr_small (clevr_query_size $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_small} (clevr_add $1 $0) $0)))"
    filter_small = f"(clevr_fold $0 clevr_empty {fold_fn})"
    
    is_yellow= "(clevr_eq_color clevr_yellow (clevr_query_color $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_yellow} (clevr_add $1 $0) $0)))"
    filter_small_yellow = f"(clevr_fold {filter_small} clevr_empty {fold_fn})"
    get_single_object = f"(clevr_car {filter_small_yellow})"
    raw_program_bootstrap = f"(lambda (clevr_query_shape {get_single_object}))"
    check_task_evaluation(zero_task, raw_program_bootstrap)
    print("\n")
    return zero_task, raw_program_original, raw_program_bootstrap
    
    
def test_zero_hop_task_query_material_original_primitives():
    zero_task = get_default_zero_hop_task_query_material()
    filter_purple = "(clevr_filter_color $0 clevr_purple)"
    get_single_object = f"(clevr_unique {filter_purple})"
    raw_program_original = f"(lambda (clevr_query_material {get_single_object}))"
    check_task_evaluation(zero_task, raw_program_original)
    print("\n")
    # Bootstrapped primitives.
    is_purple = "(clevr_eq_color clevr_purple (clevr_query_color $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_purple} (clevr_add $1 $0) $0)))"
    filter_purple = f"(clevr_fold $0 clevr_empty {fold_fn})"
    
    get_single_object = f"(clevr_car {filter_purple})"
    raw_program_bootstrap = f"(lambda (clevr_query_material {get_single_object}))"
    check_task_evaluation(zero_task, raw_program_bootstrap)
    print("\n")
    return zero_task, raw_program_original, raw_program_bootstrap
    
def test_zero_hop_task_query_color_original_primitives():
    zero_task = get_default_zero_hop_task_query_color()
    filter_metal = "(clevr_filter_material $0 clevr_metal)"
    filter_metal_sphere = f"(clevr_filter_shape {filter_metal} clevr_sphere)"
    get_single_object = f"(clevr_unique {filter_metal_sphere})"
    raw_program_original = f"(lambda (clevr_query_color {get_single_object}))"
    check_task_evaluation(zero_task, raw_program_original)
    print("\n")
    # Bootstrapped primitives.
    is_metal = "(clevr_eq_material clevr_metal (clevr_query_material $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_metal} (clevr_add $1 $0) $0)))"
    filter_metal = f"(clevr_fold $0 clevr_empty {fold_fn})"
    
    is_sphere = "(clevr_eq_shape clevr_sphere (clevr_query_shape $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_sphere} (clevr_add $1 $0) $0)))"
    filter_metal_sphere = f"(clevr_fold {filter_metal} clevr_empty {fold_fn})"
    get_single_object = f"(clevr_car {filter_metal_sphere })"
    raw_program_bootstrap = f"(lambda (clevr_query_color {get_single_object}))"
    check_task_evaluation(zero_task, raw_program_bootstrap)
    print("\n")
    return zero_task, raw_program_original, raw_program_bootstrap
    

def test_zero_hop_task_query_size_original_primitives():
    zero_task = get_default_zero_hop_task_query_size()
    filter_metal = "(clevr_filter_material $0 clevr_metal)"
    filter_brown_metal = f"(clevr_filter_color {filter_metal} clevr_brown)"
    get_single_object = f"(clevr_unique {filter_brown_metal})"
    raw_program_original = f"(lambda (clevr_query_size {get_single_object}))"
    check_task_evaluation(zero_task, raw_program_original)
    print("\n")
    # Bootstrapped primitives.
    is_metal = "(clevr_eq_material clevr_metal (clevr_query_material $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_metal} (clevr_add $1 $0) $0)))"
    filter_metal = f"(clevr_fold $0 clevr_empty {fold_fn})"
    
    is_brown = "(clevr_eq_color clevr_brown (clevr_query_color $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_brown} (clevr_add $1 $0) $0)))"
    filter_brown_metal = f"(clevr_fold {filter_metal} clevr_empty {fold_fn})"
    get_single_object = f"(clevr_car {filter_brown_metal})"
    raw_program_bootstrap = f"(lambda (clevr_query_size {get_single_object}))"
    check_task_evaluation(zero_task, raw_program_bootstrap)
    print("\n")
    return zero_task, raw_program_original, raw_program_bootstrap

def test_one_hop_count_original_primitives():
    """How many things are right the large cylinder?"""
    one_task = get_default_one_hop_count()
    filter_large = "(clevr_filter_size $0 clevr_large)"
    filter_large_cylinder = f"(clevr_filter_shape {filter_large} clevr_cylinder)"
    get_single_object = f"(clevr_unique {filter_large_cylinder})"
    get_right_of = f"(clevr_relate {get_single_object} clevr_right $0)"
    raw_program_original = f"(lambda (clevr_count {get_right_of}))"
    check_task_evaluation(one_task, raw_program_original)
    print("\n")
    # Bootstrapped primitives.
    is_large = "(clevr_eq_size clevr_large (clevr_query_size $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_large} (clevr_add $1 $0) $0)))"
    filter_large = f"(clevr_fold $0 clevr_empty {fold_fn})"
    
    is_cylinder = "(clevr_eq_shape clevr_cylinder (clevr_query_shape $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_cylinder} (clevr_add $1 $0) $0)))"
    filter_large_cylinder = f"(clevr_fold {filter_large} clevr_empty {fold_fn})"
    get_single_object = f"(clevr_car {filter_large_cylinder})"
    get_right_of = f"(clevr_relate {get_single_object} clevr_right $0)"
    raw_program_bootstrap = f"(lambda (clevr_count {get_right_of}))"
    check_task_evaluation(one_task, raw_program_bootstrap)
    print("\n")
    return one_task, raw_program_original, raw_program_bootstrap

def test_default_one_hop_query_original_primitives():
    """There is a thing front the brown thing; how big is it?"""
    one_task = get_default_one_hop_query()
    filter_brown = "(clevr_filter_color $0 clevr_brown)"
    get_single_object = f"(clevr_unique {filter_brown})"
    get_front_of = f"(clevr_relate {get_single_object} clevr_front $0)"
    get_single_object = f"(clevr_unique {get_front_of})"
    raw_program_original = f"(lambda (clevr_query_size {get_single_object}))"
    check_task_evaluation(one_task, raw_program_original)
    print("\n")
    # Bootstrapped primitives.
    is_brown = "(clevr_eq_color clevr_brown (clevr_query_color $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_brown} (clevr_add $1 $0) $0)))"
    filter_brown = f"(clevr_fold $0 clevr_empty {fold_fn})"
    get_single_object = f"(clevr_car {filter_brown})"
    get_front_of = f"(clevr_relate {get_single_object} clevr_front $0)"
    get_single_object = f"(clevr_car {get_front_of})"
    raw_program_bootstrap = f"(lambda (clevr_query_size {get_single_object}))"
    check_task_evaluation(one_task, raw_program_bootstrap)
    print("\n")
    return one_task, raw_program_original, raw_program_bootstrap
    

def test_default_same_relate_count_original_primitives():
    """How many other things are there of the same size as the cyan thing?"""
    same_relate_task = get_default_same_relate_count()
    filter_cyan = "(clevr_filter_color $0 clevr_cyan)"
    get_single_object = f"(clevr_unique {filter_cyan})"
    get_same_size = f"(clevr_same_size {get_single_object} $0)"
    raw_program_original = f"(lambda (clevr_count {get_same_size}))"
    check_task_evaluation(same_relate_task, raw_program_original)
    print("\n")
    # Bootstrapped primitives.
    is_cyan = "(clevr_eq_color clevr_cyan (clevr_query_color $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_cyan} (clevr_add $1 $0) $0)))"
    filter_cyan = f"(clevr_fold $2 clevr_empty {fold_fn})"
    get_single_object = f"(clevr_car {filter_cyan})"
    get_size = f"(clevr_query_size {get_single_object})"
    
    is_size = f"(clevr_eq_size {get_size} (clevr_query_size $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_size} (clevr_add $1 $0) $0)))"
    filter_size = f"(clevr_fold $0 clevr_empty {fold_fn})"
    
    is_cyan = "(clevr_eq_color clevr_cyan (clevr_query_color $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_cyan} (clevr_add $1 $0) $0)))"
    filter_cyan = f"(clevr_fold $0 clevr_empty {fold_fn})"
    
    difference = f"(clevr_difference {filter_size} {filter_cyan})"
    raw_program_bootstrap = f"(lambda (clevr_count {difference}))"
    check_task_evaluation(same_relate_task, raw_program_bootstrap)
    print("\n")
    return same_relate_task, raw_program_original, raw_program_bootstrap


def test_default_same_relate_query_original_primitives():
    """There is a red thing that is the same size as the metal cylinder; what shape is it?"""
    same_relate_task = get_default_same_relate_query()
    filter_metal = "(clevr_filter_material $0 clevr_metal)"
    filter_metal_cylinder = f"(clevr_filter_shape {filter_metal} clevr_cylinder)"
    get_single_object = f"(clevr_unique {filter_metal_cylinder})"
    get_same_size = f"(clevr_same_size {get_single_object} $0)"
    filter_red = f"(clevr_filter_color {get_same_size} clevr_red)"
    get_single_object = f"(clevr_unique {filter_red})"
    raw_program_original = f"(lambda (clevr_query_shape {get_single_object}))"
    check_task_evaluation(same_relate_task, raw_program_original)
    print("\n")
    is_metal = "(clevr_eq_material clevr_metal (clevr_query_material $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_metal} (clevr_add $1 $0) $0)))"
    filter_metal = f"(clevr_fold $2 clevr_empty {fold_fn})"
    
    is_cylinder = "(clevr_eq_shape clevr_cylinder (clevr_query_shape $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_cylinder} (clevr_add $1 $0) $0)))"
    filter_metal_cylinder = f"(clevr_fold {filter_metal} clevr_empty {fold_fn})"
    get_single_object = f"(clevr_unique {filter_metal_cylinder})"
    get_size = f"(clevr_query_size {get_single_object})"
    
    is_size = f"(clevr_eq_size {get_size} (clevr_query_size $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_size} (clevr_add $1 $0) $0)))"
    filter_size = f"(clevr_fold $0 clevr_empty {fold_fn})"
    
    is_red = "(clevr_eq_color clevr_red (clevr_query_color $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_red} (clevr_add $1 $0) $0)))"
    filter_red = f"(clevr_fold {filter_size} clevr_empty {fold_fn})"
    get_single_object = f"(clevr_unique {filter_red})"
    raw_program_bootstrap = f"(lambda (clevr_query_shape {get_single_object}))"
    check_task_evaluation(same_relate_task, raw_program_bootstrap)
    print("\n")
    return same_relate_task, raw_program_original, raw_program_bootstrap
    
    
def test_default_compare_integer_less_than_original_primitives():
    """Is the number of cyan rubber things less than the number of large cylinders?"""
    compare_integer_task = get_default_compare_integer_less_than()
    filter_large = "(clevr_filter_size $0 clevr_large)"
    filter_large_cylinder = f"(clevr_filter_shape {filter_large} clevr_cylinder)"
    filter_rubber = "(clevr_filter_material $0 clevr_rubber)"
    filter_cyan_rubber = f"(clevr_filter_color {filter_rubber} clevr_cyan)"
    raw_program_original = f"(lambda (clevr_lt? (clevr_count {filter_cyan_rubber}) (clevr_count {filter_large_cylinder})))"
    check_task_evaluation(compare_integer_task, raw_program_original)
    print("\n")
    # Bootstrap primitives
    is_cyan = "(clevr_eq_color clevr_cyan (clevr_query_color $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_cyan} (clevr_add $1 $0) $0)))"
    filter_cyan = f"(clevr_fold $0 clevr_empty {fold_fn})"
    is_rubber = "(clevr_eq_material clevr_rubber (clevr_query_material $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_rubber} (clevr_add $1 $0) $0)))"
    filter_cyan_rubber = f"(clevr_fold {filter_cyan} clevr_empty {fold_fn})"
    
    is_large = "(clevr_eq_size clevr_large (clevr_query_size $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_large} (clevr_add $1 $0) $0)))"
    filter_large = f"(clevr_fold $0 clevr_empty {fold_fn})"
    is_cylinder = "(clevr_eq_shape clevr_cylinder (clevr_query_shape $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_cylinder} (clevr_add $1 $0) $0)))"
    filter_large_cylinder = f"(clevr_fold {filter_large} clevr_empty {fold_fn})"
    
    raw_program_bootstrap = f"(lambda (clevr_lt? (clevr_count {filter_cyan_rubber}) (clevr_count {filter_large_cylinder})))"
    check_task_evaluation(compare_integer_task, raw_program_bootstrap)
    print("\n")
    return compare_integer_task, raw_program_original, raw_program_bootstrap

def test_default_compare_integer_greater_than():
    """Is the number of large cylinders greater than the number of small rubber spheres?"""
    compare_integer_task = get_default_compare_integer_greater_than()
    filter_large = "(clevr_filter_size $0 clevr_large)"
    filter_large_cylinder = f"(clevr_filter_shape {filter_large} clevr_cylinder)"
    filter_small = "(clevr_filter_size $0 clevr_small)"
    filter_small_sphere = f"(clevr_filter_shape {filter_small} clevr_sphere)"
    filter_small_rubber_sphere = f"(clevr_filter_material {filter_small_sphere} clevr_rubber)"
    
    raw_program_original = f"(lambda (clevr_gt? (clevr_count {filter_large_cylinder}) (clevr_count {filter_small_rubber_sphere})))"
    check_task_evaluation(compare_integer_task, raw_program_original)
    print("\n")
    
    # Bootstrap primitives
    is_small = "(clevr_eq_size clevr_small (clevr_query_size $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_small} (clevr_add $1 $0) $0)))"
    filter_small = f"(clevr_fold $0 clevr_empty {fold_fn})"
    
    is_rubber = "(clevr_eq_material clevr_rubber (clevr_query_material $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_rubber} (clevr_add $1 $0) $0)))"
    filter_small_rubber = f"(clevr_fold {filter_small} clevr_empty {fold_fn})"
    is_sphere = "(clevr_eq_shape clevr_sphere (clevr_query_shape $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_sphere} (clevr_add $1 $0) $0)))"
    filter_small_rubber_sphere = f"(clevr_fold {filter_small_rubber} clevr_empty {fold_fn})"
    
    is_large = "(clevr_eq_size clevr_large (clevr_query_size $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_large} (clevr_add $1 $0) $0)))"
    filter_large = f"(clevr_fold $0 clevr_empty {fold_fn})"
    is_cylinder = "(clevr_eq_shape clevr_cylinder (clevr_query_shape $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_cylinder} (clevr_add $1 $0) $0)))"
    filter_large_cylinder = f"(clevr_fold {filter_large} clevr_empty {fold_fn})"
    
    raw_program_bootstrap = f"(lambda (clevr_gt? (clevr_count {filter_large_cylinder}) (clevr_count {filter_small_rubber_sphere})))"
    check_task_evaluation(compare_integer_task, raw_program_bootstrap)
    print("\n")
    return compare_integer_task, raw_program_original, raw_program_bootstrap

def test_default_single_or():
    """How many cylinders are brown things or small rubber things?"""
    single_or_task = get_default_single_or()
    filter_brown = f"(clevr_filter_color $0 clevr_brown)"
    filter_rubber = f"(clevr_filter_material $0 clevr_rubber)"
    filter_small_rubber = f"(clevr_filter_size {filter_rubber} clevr_small)"
    union = f"(clevr_union {filter_brown} {filter_small_rubber})"
    filter_cylinders = f"(clevr_filter_shape {union} clevr_cylinder)"
    raw_program_original = f"(lambda (clevr_count {filter_cylinders}))"
    check_task_evaluation(single_or_task, raw_program_original)
    print("\n")
    # Bootstrap primitives
    is_small = "(clevr_eq_size clevr_small (clevr_query_size $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_small} (clevr_add $1 $0) $0)))"
    filter_small = f"(clevr_fold $0 clevr_empty {fold_fn})"
    
    is_rubber = "(clevr_eq_material clevr_rubber (clevr_query_material $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_rubber} (clevr_add $1 $0) $0)))"
    filter_small_rubber = f"(clevr_fold {filter_small} clevr_empty {fold_fn})"
    
    is_brown = "(clevr_eq_color clevr_brown (clevr_query_color $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_brown} (clevr_add $1 $0) $0)))"
    filter_brown = f"(clevr_fold $0 clevr_empty {fold_fn})"
    union = f"(clevr_union {filter_brown} {filter_small_rubber})"
    
    is_cylinder = "(clevr_eq_shape clevr_cylinder (clevr_query_shape $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_cylinder} (clevr_add $1 $0) $0)))"
    filter_cylinder = f"(clevr_fold {union} clevr_empty {fold_fn})"
    raw_program_bootstrap = f"(lambda (clevr_count {filter_cylinder}))"
    check_task_evaluation(single_or_task, raw_program_bootstrap)
    print("\n")
    return single_or_task, raw_program_original, raw_program_bootstrap

def test_default_remove():
    """What if you removed all of the blue metal things?"""
    remove_task = get_default_remove()
    filter_blue = f"(clevr_filter_color $0 clevr_blue)"
    filter_blue_metal = f"(clevr_filter_material {filter_blue} clevr_metal)"
    raw_program_original = f"(lambda (clevr_difference $0 {filter_blue_metal}))"
    check_task_evaluation(remove_task, raw_program_original)
    print("\n")
    is_blue = "(clevr_eq_color clevr_blue (clevr_query_color $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_blue} (clevr_add $1 $0) $0)))"
    filter_blue = f"(clevr_fold $0 clevr_empty {fold_fn})"
    
    is_metal = "(clevr_eq_material clevr_metal (clevr_query_material $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_metal} (clevr_add $1 $0) $0)))"
    filter_blue_metal = f"(clevr_fold {filter_blue} clevr_empty {fold_fn})"
    raw_program_bootstrap = f"(lambda (clevr_difference $0 {filter_blue_metal}))"
    check_task_evaluation(remove_task, raw_program_bootstrap)
    print("\n")
    return remove_task, raw_program_original, raw_program_bootstrap

def test_default_remove_query():
    """If you removed the red things, how many spheres would be left?"""
    remove_task = get_default_remove_query()
    filter_red = f"(clevr_filter_color $0 clevr_red)"
    remove = f"(clevr_difference $0 {filter_red})"
    filter_spheres = f"(clevr_filter_shape {remove} clevr_sphere)"
    raw_program_original = f"(lambda (clevr_count {filter_spheres}))"
    check_task_evaluation(remove_task, raw_program_original)
    print("\n")
    is_red = "(clevr_eq_color clevr_red (clevr_query_color $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_red} (clevr_add $1 $0) $0)))"
    filter_red = f"(clevr_fold $0 clevr_empty {fold_fn})"
    remove = f"(clevr_difference $0 {filter_red})"
    
    is_sphere = "(clevr_eq_shape clevr_sphere (clevr_query_shape $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_sphere} (clevr_add $1 $0) $0)))"
    filter_spheres = f"(clevr_fold {remove} clevr_empty {fold_fn})"
    raw_program_bootstrap = f"(lambda (clevr_count {filter_spheres}))"
    check_task_evaluation(remove_task, raw_program_bootstrap)
    print("\n")
    return remove_task, raw_program_original, raw_program_bootstrap

def test_default_transform():
    """What if the gray sphere became a small green metal sphere?"""
    transform_task = get_default_transform()
    filter_gray = f"(clevr_filter_color $0 clevr_gray)"
    filter_gray_sphere = f"(clevr_filter_shape {filter_gray} clevr_sphere)"
    single_object = f"(clevr_unique {filter_gray_sphere})"
    transform_small = f"(clevr_transform_size clevr_small {single_object})"
    transform_small_green = f"(clevr_transform_color clevr_green {transform_small})"
    transform_small_green_metal = f"(clevr_transform_material clevr_metal {transform_small_green})"
    raw_program_original = f"(lambda (clevr_add {transform_small_green_metal} $0))"
    check_task_evaluation(transform_task, raw_program_original)
    print("\n")
    # Bootstrap primitives
    is_gray = "(clevr_eq_color clevr_gray (clevr_query_color $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_gray} (clevr_add $1 $0) $0)))"
    filter_gray = f"(clevr_fold $0 clevr_empty {fold_fn})"
    is_sphere = "(clevr_eq_shape clevr_sphere (clevr_query_shape $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_sphere} (clevr_add $1 $0) $0)))"
    filter_gray_sphere = f"(clevr_fold {filter_gray} clevr_empty {fold_fn})"
    
    map_transform_green = f"(clevr_map (clevr_transform_color clevr_green) {filter_gray_sphere})"
    map_transform_green_metal = f"(clevr_map (clevr_transform_material clevr_metal) {map_transform_green})"
    map_transform_small_green_metal = f"(clevr_map (clevr_transform_size clevr_small) {map_transform_green_metal})"
    raw_program_bootstrap = f"(lambda (clevr_union {map_transform_small_green_metal} $0))"
    check_task_evaluation(transform_task, raw_program_bootstrap)
    print("\n")
    return transform_task, raw_program_original, raw_program_bootstrap
    

def test_default_transform_query():
    """If all of the blue spheres became brown, how many brown things would there be?"""
    transform_task = get_default_transform_query()
    filter_blue = f"(clevr_filter_color $0 clevr_blue)"
    filter_blue_sphere = f"(clevr_filter_shape {filter_blue} clevr_sphere)"
    map_transform = f"(clevr_map (clevr_transform_color clevr_brown) {filter_blue_sphere})"
    union = f"(clevr_union {map_transform} $0)"
    filter_brown = f"(clevr_filter_color {union} clevr_brown)"
    raw_program_original = f"(lambda (clevr_count {filter_brown}))"
    check_task_evaluation(transform_task, raw_program_original)
    print("\n")
    is_blue = "(clevr_eq_color clevr_blue (clevr_query_color $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_blue} (clevr_add $1 $0) $0)))"
    filter_gray = f"(clevr_fold $0 clevr_empty {fold_fn})"
    is_sphere = "(clevr_eq_shape clevr_sphere (clevr_query_shape $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_sphere} (clevr_add $1 $0) $0)))"
    filter_blue_sphere = f"(clevr_fold {filter_blue} clevr_empty {fold_fn})"
    map_transform = f"(clevr_map (clevr_transform_color clevr_brown) {filter_blue_sphere})"
    union = f"(clevr_union {map_transform} $0)"
    
    is_brown = "(clevr_eq_color clevr_brown (clevr_query_color $1))"
    fold_fn = f"(lambda (lambda (clevr_if {is_brown} (clevr_add $1 $0) $0)))"
    filter_brown = f"(clevr_fold {union} clevr_empty {fold_fn})"
    raw_program_bootstrap = f"(lambda (clevr_count {filter_brown}))"
    check_task_evaluation(transform_task, raw_program_bootstrap)
    print("\n")
    return transform_task, raw_program_original, raw_program_bootstrap

def test_all():
    print("Running tests for clevrPrimitives....")
    
    # Tests for common error scenarios.
    # test_relate_not_in_list()
    # test_relate_no_relations()
    # test_union_same_lists()
    # test_intersect_no_intersection()
    # test_difference_one_empty_list()
    # test_difference_empty_lists()
    # test_add_duplicate()
    # test_car_empty_list()
    # test_if_malformed()
    # test_map_malformed()
    # test_map_malformed_transform_only()
    # test_fold_malformed()
    # Tests that should succeed
    # test_localization_task_original_primitives()
    # test_localization_task_original_primitives_base_filter()
    # test_localization_task_multiple_filter_original_primitives()
    # test_zero_hop_task_count_original_primitives()
    # test_zero_hop_task_query_shape_original_primitives()
    # test_zero_hop_task_query_material_original_primitives()
    # test_zero_hop_task_query_color_original_primitives()
    # test_zero_hop_task_query_size_original_primitives()
    # test_one_hop_count_original_primitives()
    # test_default_one_hop_query_original_primitives()
    # 
    # test_default_same_relate_count_original_primitives()
    # test_default_same_relate_query_original_primitives() 
    # 
    # test_default_compare_integer_less_than_original_primitives()
    # test_default_compare_integer_greater_than()
    # test_default_single_or()
    # test_default_remove()
    # test_default_remove_query()
    test_default_transform()
    # test_default_transform_query()
    # # 
    # pass