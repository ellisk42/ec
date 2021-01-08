"""
Test for clevrSolver.ml | Author : Catherine Wong
Tests for the CLEVR OCaml implementations of primitives and task handling. 

All tests are manually added to a 'test_all' function.

The task-based tests using a raw program string import tasks and program strings written in test_clevrPrimitives.py. 
    These tests return a (task, raw_program_original, raw_program_bootstrap) tuple.
"""
import dreamcoder.domains.clevr.test_clevrPrimitives as test_clevrPrimitives

from dreamcoder.grammar import * 
from dreamcoder.enumeration import solveForTask_ocaml, multicoreEnumeration
import dreamcoder.domains.clevr.clevrPrimitives as clevrPrimitives

CLEVR_PRIMITIVE_OCAML_TEST_FILE = "test_clevr_primitives"
CLEVR_SOLVER_OCAML_FILE = "clevrSolver"

CLEVR_PRIMITIVE_SETS = ['clevr_original', 'clevr_bootstrap', 'clevr_map_transform']
CLEVR_ORIGINAL_PRIMITIVES = ['clevr_original', 'clevr_map_transform']
CLEVR_BOOTSTRAP_PRIMITIVES = [ 'clevr_bootstrap', 'clevr_map_transform']
EVALUTION_TIMEOUT = 0.5

def check_ocaml_evaluation_for_task(task, programs_to_test, should_succeed=True):
    """Helper method to evaluate a specific program"""
    TIMEOUT = 10
    # Construct a grammar object.
    clevr_primitives = clevrPrimitives.load_clevr_primitives(CLEVR_PRIMITIVE_SETS)
    clevr_grammar = Grammar.uniform(clevr_primitives)
    
    # Add the desired programs to the task to be tested.
    task.raw_programs_to_test = programs_to_test
    # Set the special solver to the testing file.
    task.specialSolver = CLEVR_PRIMITIVE_OCAML_TEST_FILE
    response = solveForTask_ocaml(g=clevr_grammar,
                       unigramGrammar=clevr_grammar,
                       maximumFrontiers = {task : 1},
                       tasks=[task],
                       evaluationTimeout=TIMEOUT,
                       lowerBound=0,
                       upperBound=100,
                       budgetIncrement=0.5,
                       verbose=True
                       )
    print(response)
    assert task.name in response
    did_succeed =  response[task.name]
    assert did_succeed == should_succeed

def check_ocaml_enumeration_for_single_thread(tasks, primitives_set, timeout, should_succeed=None):
    """Helper method to test actual enumeration on a single program."""
    # Construct a grammar object.
    clevr_primitives = clevrPrimitives.load_clevr_primitives(primitives_set)
    clevr_grammar = Grammar.uniform(clevr_primitives)
    MAX_FRONTIERS = 1
    max_frontiers_per_task = {task : MAX_FRONTIERS for task in tasks}
    # Set the special solver to the testing file.
    for task in tasks:
        task.specialSolver = CLEVR_SOLVER_OCAML_FILE
    frontiers, searchTimes, number_enumerated = solveForTask_ocaml(g=clevr_grammar,
                       unigramGrammar=clevr_grammar,
                       maximumFrontiers = max_frontiers_per_task,
                       tasks=tasks,
                       timeout=float(timeout),
                       evaluationTimeout=float(EVALUTION_TIMEOUT),
                       lowerBound=0,
                       upperBound=100,
                       budgetIncrement=0.5,
                       verbose=True
                       )
    assert number_enumerated > 0
    for task_idx, task in enumerate(tasks):
        assert task in frontiers
        frontier_for_task = frontiers[task]
        assert frontier_for_task.task.name == task.name
        assert task in searchTimes
        
        if should_succeed is None or should_succeed[task_idx]:
            assert not frontier_for_task.empty
            for entry in frontier_for_task.entries:
                assert len(entry.tokens) > 0
            assert len(frontier_for_task.entries) <= MAX_FRONTIERS
        else:
            assert frontier_for_task.empty

def check_ocaml_enumeration_for_multi_thread(tasks, primitives_set, timeout, should_succeed=None, check_success=True):
    # Construct a grammar object.
    clevr_primitives = clevrPrimitives.load_clevr_primitives(primitives_set)
    clevr_grammar = Grammar.uniform(clevr_primitives)
    MAX_FRONTIERS = 1
    # Set the special solver to the testing file.
    for task in tasks:
        task.specialSolver = CLEVR_SOLVER_OCAML_FILE
    frontiers_per_task, bestSearchTime = multicoreEnumeration(clevr_grammar, tasks, _=None,
                             enumerationTimeout=float(timeout),
                             solver='ocaml',
                             CPUs=1,
                             maximumFrontier=MAX_FRONTIERS,
                             verbose=True,
                             evaluationTimeout=float(EVALUTION_TIMEOUT),
                             testing=False,
                             unigramGrammar=clevr_grammar)
    for task_idx, task in enumerate(tasks):
        frontier_for_task = frontiers_per_task[task_idx]
        assert frontier_for_task.task.name == task.name
        assert task in bestSearchTime
        
        if not check_success: return
        if should_succeed is None or should_succeed[task_idx]:
            assert not frontier_for_task.empty
            for entry in frontier_for_task.entries:
                assert len(entry.tokens) > 0
            assert len(frontier_for_task.entries) <= MAX_FRONTIERS
        else:
            assert frontier_for_task.empty

# Integration stress testing.
def test_enumeration_localization_task_original_primitives():
    TIMEOUT = 10
    task, _, _ = test_clevrPrimitives.test_localization_task_original_primitives()
    check_ocaml_enumeration_for_single_thread(task, primitives_set=CLEVR_ORIGINAL_PRIMITIVES, timeout=TIMEOUT, should_succeed=True)

def test_enumeration_localization_bootstrap_primitives():
    TIMEOUT = 10
    task, _, _ = test_clevrPrimitives.test_localization_task_original_primitives()
    check_ocaml_enumeration_for_single_thread(task, primitives_set=CLEVR_BOOTSTRAP_PRIMITIVES, timeout=TIMEOUT, should_succeed=False)

def test_enumeration_one_hop_original_primitives():
    TIMEOUT = 10
    task, _, _ = test_clevrPrimitives.test_default_one_hop_query_original_primitives()
    check_ocaml_enumeration_for_single_thread(task, primitives_set=CLEVR_ORIGINAL_PRIMITIVES, timeout=TIMEOUT, should_succeed=False)

def test_enumeration_one_hop_bootstrap_primitives():
    TIMEOUT = 10
    task, _, _ = test_clevrPrimitives.test_one_hop_count_original_primitives()
    check_ocaml_enumeration_for_single_thread(task, primitives_set=CLEVR_ORIGINAL_PRIMITIVES, timeout=TIMEOUT, should_succeed=False)

def get_same_type_batch_of_tasks_scene():
    """Gets a batch of tasks with scene_return_types"""
    task_localization, _, _ = test_clevrPrimitives.test_localization_task_original_primitives()
    task_localization_multiple, _, _ = test_clevrPrimitives.test_localization_task_multiple_filter_original_primitives()
    task_remove, _, _ = test_clevrPrimitives.test_default_remove()
    return [task_remove, task_localization, task_localization_multiple]

def get_same_type_batch_of_tasks_int():
    task_relate_int, _, _ = test_clevrPrimitives.test_default_same_relate_count_original_primitives()
    task_count_int, _, _ = test_clevrPrimitives.test_one_hop_count_original_primitives()
    return [task_relate_int, task_count_int]
        
def get_mixed_type_batch_of_tasks():
    """Gets a batch of tasks with mixed return types"""
    task_localization, _, _ = test_clevrPrimitives.test_localization_task_original_primitives()
    task_count, _, _ = test_clevrPrimitives.test_zero_hop_task_count_original_primitives()
    task_size, _, _ = test_clevrPrimitives.test_zero_hop_task_query_size_original_primitives()
    task_bool, _, _ = test_clevrPrimitives.test_default_compare_integer_greater_than()
    task_remove, _, _ = test_clevrPrimitives.test_default_remove()
    return [task_localization, task_count, task_size, task_bool, task_remove]

def get_mixed_type_batch_of_tasks_all_different_types():
    """Gets a batch of tasks with mixed return types, but each is different -- no Python parallelism."""
    task_localization, _, _ = test_clevrPrimitives.test_localization_task_original_primitives()
    task_count, _, _ = test_clevrPrimitives.test_zero_hop_task_count_original_primitives()
    task_size, _, _ = test_clevrPrimitives.test_zero_hop_task_query_size_original_primitives()
    task_bool, _, _ = test_clevrPrimitives.test_default_compare_integer_greater_than()
    return [task_localization, task_count, task_size, task_bool]
    
def test_enumeration_single_thread_multiple_task_scene_original_primitives():
    tasks = get_same_type_batch_of_tasks_scene()
    TIMEOUT = 10
    check_ocaml_enumeration_for_single_thread(tasks, primitives_set=CLEVR_ORIGINAL_PRIMITIVES, timeout=TIMEOUT, should_succeed=[False, True, True])

def test_enumeration_single_thread_multiple_task_scene_bootstrap_primitives():
    tasks = get_same_type_batch_of_tasks_scene()
    TIMEOUT = 10
    check_ocaml_enumeration_for_single_thread(tasks, primitives_set=CLEVR_BOOTSTRAP_PRIMITIVES, timeout=TIMEOUT, should_succeed=[False, False, False])

def test_enumeration_single_thread_multiple_task_scene_original_primitives():
    tasks = get_same_type_batch_of_tasks_int()
    TIMEOUT = 10
    check_ocaml_enumeration_for_single_thread(tasks, primitives_set=CLEVR_ORIGINAL_PRIMITIVES, timeout=TIMEOUT, should_succeed=[True, False])

def test_enumeration_single_thread_multiple_task_scene_bootstrap_primitives():
    tasks = get_same_type_batch_of_tasks_int()
    TIMEOUT = 10
    check_ocaml_enumeration_for_single_thread(tasks, primitives_set=CLEVR_BOOTSTRAP_PRIMITIVES, timeout=TIMEOUT, should_succeed=[False, False])
    
# Tests parallel enumeration for a set of tasks.
def test_parallel_enumeration_mini_batch_original_primitives():
    TIMEOUT = 10
    tasks = get_mixed_type_batch_of_tasks()
    check_ocaml_enumeration_for_multi_thread(tasks, primitives_set=CLEVR_ORIGINAL_PRIMITIVES, timeout=TIMEOUT, should_succeed=None, check_success=False)

def test_parallel_enumeration_mini_batch_bootstrap_primitives():
    TIMEOUT = 10
    tasks = get_mixed_type_batch_of_tasks()
    check_ocaml_enumeration_for_multi_thread(tasks, primitives_set=CLEVR_BOOTSTRAP_PRIMITIVES, timeout=TIMEOUT, should_succeed=None, check_success=False)

# Tests how we run Helmholtz samples.

# Tests error handling on common error scenarios.
def test_relate_not_in_list_ocaml():
    task, error_program = test_clevrPrimitives.test_relate_not_in_list()
    check_ocaml_evaluation_for_task(task, [error_program], should_succeed=False)
    print("\n")
def test_relate_no_relations_ocaml():
    task, error_program = test_clevrPrimitives.test_relate_no_relations()
    check_ocaml_evaluation_for_task(task, [error_program], should_succeed=False)
    print("\n")
def test_union_same_lists_ocaml():
    task, double_union = test_clevrPrimitives.test_union_same_lists()
    check_ocaml_evaluation_for_task(task, [double_union])
    print("\n")
def test_intersect_no_intersection_ocaml():
    task, intersection = test_clevrPrimitives.test_intersect_no_intersection()
    check_ocaml_evaluation_for_task(task, [intersection])
    print("\n")
def test_difference_empty_lists_ocaml():
    task, difference = test_clevrPrimitives.test_difference_empty_lists()
    check_ocaml_evaluation_for_task(task, [difference])
    print("\n")
def test_difference_one_empty_list_ocaml():
    task, difference = test_clevrPrimitives.test_difference_one_empty_list()
    check_ocaml_evaluation_for_task(task, [difference])
    print("\n")
def test_car_empty_list_ocaml():
    task, error_program = test_clevrPrimitives.test_car_empty_list()
    check_ocaml_evaluation_for_task(task, [error_program], should_succeed=False)
    print("\n")
def test_add_duplicate_ocaml():
    task, duplicate = test_clevrPrimitives.test_add_duplicate()
    check_ocaml_evaluation_for_task(task, [duplicate])
    print("\n")
def test_if_malformed_ocaml():
    task, error_program = test_clevrPrimitives.test_if_malformed()
    check_ocaml_evaluation_for_task(task, [error_program], should_succeed=False)
    print("\n")
def test_map_malformed_ocaml():
    task, error_program = test_clevrPrimitives.test_map_malformed()
    check_ocaml_evaluation_for_task(task, [error_program], should_succeed=False)
    print("\n")
def test_map_malformed_transform_only_ocaml():
    task, error_program = test_clevrPrimitives.test_map_malformed_transform_only()
    check_ocaml_evaluation_for_task(task, [error_program], should_succeed=False)
    print("\n")
def test_fold_malformed_ocaml():
    task, error_program = test_clevrPrimitives.test_fold_malformed()
    check_ocaml_evaluation_for_task(task, [error_program], should_succeed=False)
    print("\n")
    
# Tests that we can solve each of the individual question classes with the reimplementation.
def test_localization_task_ocaml():
    print(f"Running test_localization_task_ocaml")
    task, raw_program_original, raw_program_bootstrap = test_clevrPrimitives.test_localization_task_original_primitives()
    check_ocaml_evaluation_for_task(task, [raw_program_original, raw_program_bootstrap])
    print("\n")

def test_localizatization_task_base_filter_ocaml():
    print(f"Running test_localizatization_task_base_filter_ocaml")
    localization_task, raw_program = test_clevrPrimitives.test_localization_task_original_primitives_base_filter()
    check_ocaml_evaluation_for_task(localization_task, [raw_program])
    print("\n")

def test_localization_task_multiple_filter_original_primitives_ocaml():
    print ("Running test_localization_task_multiple_filter_original_primitives_ocaml")
    task, raw_program_original, raw_program_bootstrap = test_clevrPrimitives.test_localization_task_multiple_filter_original_primitives()
    check_ocaml_evaluation_for_task(task, [raw_program_original, raw_program_bootstrap])
    print("\n")

def test_zero_hop_task_count_ocaml():
    print ("Running test_zero_hop_task_count_ocaml")
    task, raw_program_original, raw_program_bootstrap = test_clevrPrimitives.test_zero_hop_task_count_original_primitives()
    check_ocaml_evaluation_for_task(task, [raw_program_original, raw_program_bootstrap])
    print("\n")

def test_zero_hop_task_query_shape_ocaml():
    print("Running test_zero_hop_task_query_shape_ocaml")
    task, raw_program_original, raw_program_bootstrap = test_clevrPrimitives.test_zero_hop_task_query_shape_original_primitives()
    check_ocaml_evaluation_for_task(task, [raw_program_original, raw_program_bootstrap])
    print("\n")

def test_zero_hop_task_query_material_ocaml():
    print("Running test_zero_hop_task_query_material_ocaml")
    task, raw_program_original, raw_program_bootstrap = test_clevrPrimitives.test_zero_hop_task_query_material_original_primitives()
    check_ocaml_evaluation_for_task(task, [raw_program_original, raw_program_bootstrap])
    print("\n")

def test_zero_hop_task_query_color_ocaml():
    print("Running test_zero_hop_task_query_color_ocaml")
    task, raw_program_original, raw_program_bootstrap = test_clevrPrimitives.test_zero_hop_task_query_color_original_primitives()
    check_ocaml_evaluation_for_task(task, [raw_program_original, raw_program_bootstrap])
    print("\n")

def test_zero_hop_task_query_size_ocaml():
    print("Running test_zero_hop_task_query_size_ocaml")
    task, raw_program_original, raw_program_bootstrap = test_clevrPrimitives.test_zero_hop_task_query_size_original_primitives()
    check_ocaml_evaluation_for_task(task, [raw_program_original, raw_program_bootstrap])
    print("\n")

def test_one_hop_count_ocaml():
    print("Running test_one_hop_count_ocaml")
    task, raw_program_original, raw_program_bootstrap = test_clevrPrimitives.test_one_hop_count_original_primitives()
    check_ocaml_evaluation_for_task(task, [raw_program_original, raw_program_bootstrap])
    print("\n")

def test_default_one_hop_query_ocaml():
    print("Running test_default_one_hop_query_ocaml")
    task, raw_program_original, raw_program_bootstrap = test_clevrPrimitives.test_default_one_hop_query_original_primitives()
    check_ocaml_evaluation_for_task(task, [raw_program_original, raw_program_bootstrap])
    print("\n")

def test_default_same_relate_count_ocaml():
    print("Running test_default_same_relate_count_ocaml")
    task, raw_program_original, raw_program_bootstrap = test_clevrPrimitives.test_default_same_relate_count_original_primitives()
    check_ocaml_evaluation_for_task(task, [raw_program_original, raw_program_bootstrap])
    print("\n")

def test_default_same_relate_query_ocaml():
    print("Running test_default_same_relate_query_ocaml")
    task, raw_program_original, raw_program_bootstrap = test_clevrPrimitives.test_default_same_relate_query_original_primitives()
    check_ocaml_evaluation_for_task(task, [raw_program_original, raw_program_bootstrap])
    print("\n")

def test_default_compare_integer_less_than_ocaml():
    print("Running test_default_compare_integer_less_than_ocaml")
    task, raw_program_original, raw_program_bootstrap = test_clevrPrimitives.test_default_compare_integer_less_than_original_primitives()
    check_ocaml_evaluation_for_task(task, [raw_program_original, raw_program_bootstrap])
    print("\n")

def test_default_compare_integer_greater_than_ocaml():
    print("Running test_default_compare_integer_greater_than_ocaml")
    task, raw_program_original, raw_program_bootstrap = test_clevrPrimitives.test_default_compare_integer_greater_than()
    check_ocaml_evaluation_for_task(task, [raw_program_original, raw_program_bootstrap])
    print("\n")

def test_default_single_or_ocaml():
    print("Running test_default_single_or_ocaml")
    task, raw_program_original, raw_program_bootstrap = test_clevrPrimitives.test_default_single_or()
    check_ocaml_evaluation_for_task(task, [raw_program_original, raw_program_bootstrap])
    print("\n")
    
def test_default_remove_ocaml():
    print("Running test_default_remove_ocaml")
    task, raw_program_original, raw_program_bootstrap = test_clevrPrimitives.test_default_remove()
    check_ocaml_evaluation_for_task(task, [raw_program_original, raw_program_bootstrap])
    print("\n")

def test_default_remove_query_ocaml():
    print("Running test_default_remove_query_ocaml")
    task, raw_program_original, raw_program_bootstrap = test_clevrPrimitives.test_default_remove_query()
    check_ocaml_evaluation_for_task(task, [raw_program_original, raw_program_bootstrap])
    print("\n")


def test_default_transform_ocaml():
    print("Running test_default_transform_ocaml")
    task, raw_program_original, raw_program_bootstrap = test_clevrPrimitives.test_default_transform()
    check_ocaml_evaluation_for_task(task, [raw_program_original, raw_program_bootstrap])
    print("\n")

def test_default_transform_query_ocaml():
    print("Running test_default_transform_query_ocaml")
    task, raw_program_original, raw_program_bootstrap = test_clevrPrimitives.test_default_transform_query()
    check_ocaml_evaluation_for_task(task, [raw_program_original, raw_program_bootstrap])
    print("\n")

    
def test_all():
    print("Running tests for clevrPrimitivesOcaml....")
    # Error handling programs.
    test_relate_not_in_list_ocaml()
    test_relate_no_relations_ocaml()
    test_union_same_lists_ocaml()
    test_intersect_no_intersection_ocaml()
    test_difference_empty_lists_ocaml()
    test_difference_one_empty_list_ocaml()
    test_add_duplicate_ocaml()
    test_car_empty_list_ocaml()
    test_if_malformed_ocaml()
    test_map_malformed_ocaml()
    test_map_malformed_transform_only_ocaml()
    test_fold_malformed_ocaml()
    
    # Individual task classes
    test_localization_task_ocaml()
    test_localizatization_task_base_filter_ocaml()
    test_localization_task_multiple_filter_original_primitives_ocaml()
    test_zero_hop_task_count_ocaml()
    test_zero_hop_task_query_shape_ocaml()
    test_zero_hop_task_query_material_ocaml()
    test_zero_hop_task_query_color_ocaml()
    test_zero_hop_task_query_size_ocaml()
    test_one_hop_count_ocaml()
    test_default_one_hop_query_ocaml()
    test_default_same_relate_count_ocaml()
    test_default_same_relate_query_ocaml()
    test_default_compare_integer_less_than_ocaml()
    test_default_compare_integer_greater_than_ocaml()
    test_default_single_or_ocaml()
    test_default_remove_ocaml()
    test_default_remove_query_ocaml()
    test_default_transform_ocaml()
    test_default_transform_query_ocaml()
    
    # Enumeration
    test_enumeration_localization_task_original_primitives()
    test_enumeration_localization_bootstrap_primitives()
    test_enumeration_one_hop_original_primitives()
    test_enumeration_one_hop_bootstrap_primitives()
    test_enumeration_single_thread_multiple_task_scene_original_primitives()
    test_enumeration_single_thread_multiple_task_scene_bootstrap_primitives()
    test_enumeration_single_thread_multiple_task_scene_original_primitives()
    test_enumeration_single_thread_multiple_task_scene_bootstrap_primitives()
    test_parallel_enumeration_mini_batch_original_primitives()
    test_parallel_enumeration_mini_batch_bootstrap_primitives()
    print(".....finished running all tests!")
