"""
Test for clevrSolver.ml | Author : Catherine Wong
Tests for the CLEVR OCaml implementations of primitives and task handling. 

All tests are manually added to a 'test_all' function.

The task-based tests using a raw program string import tasks and program strings written in test_clevrPrimitives.py. 
    These tests return a (task, raw_program_original, raw_program_bootstrap) tuple.
"""
import dreamcoder.domains.clevr.test_clevrPrimitives as test_clevrPrimitives

from dreamcoder.grammar import * 
from dreamcoder.enumeration import solveForTask_ocaml
import dreamcoder.domains.clevr.clevrPrimitives as clevrPrimitives

CLEVR_PRIMITIVE_OCAML_TEST_FILE = "test_clevr_primitives"

CLEVR_PRIMITIVE_SETS = ['clevr_original', 'clevr_bootstrap', 'clevr_map_transform']

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


# Integration stress testing.
def test_enumeration_localization_task_original_primitives():
    pass

def test_enumeration_localization_bootstrap_primitives():
    pass

def test_enumeration_one_hop_original_primitives():
    pass

def test_enumeration_one_hop_bootstrap_primitives():
    pass

# Tests parallel enumeration for a set of tasks.
def test_parallel_enumeration_mini_batch_original_primitives():
    pass

def test_parallel_enumeration_mini_batch_bootstrap_primitives():
    pass

# Tests how we run Helmholtz samples.

    
def test_all():
    print("Running tests for clevrPrimitivesOcaml....")
    # Error handling programs.
    # test_relate_not_in_list_ocaml()
    # test_relate_no_relations_ocaml()
    # test_union_same_lists_ocaml()
    # test_intersect_no_intersection_ocaml()
    # test_difference_empty_lists_ocaml()
    # test_difference_one_empty_list_ocaml()
    # test_add_duplicate_ocaml()
    # test_car_empty_list_ocaml()
    # test_if_malformed_ocaml()
    # test_map_malformed_ocaml()
    # test_map_malformed_transform_only_ocaml()
    test_fold_malformed_ocaml()
    
    # test_localization_task_ocaml()
    # test_localizatization_task_base_filter_ocaml()
    # test_localization_task_multiple_filter_original_primitives_ocaml()
    # test_zero_hop_task_count_ocaml()
    # test_zero_hop_task_query_shape_ocaml()
    # test_zero_hop_task_query_material_ocaml()
    # test_zero_hop_task_query_color_ocaml()
    # test_zero_hop_task_query_size_ocaml()
    # test_one_hop_count_ocaml()
    # test_default_one_hop_query_ocaml()
    # test_default_same_relate_count_ocaml()
    # test_default_same_relate_query_ocaml()
    # test_default_compare_integer_less_than_ocaml()
    # test_default_compare_integer_greater_than_ocaml()
    # test_default_single_or_ocaml()
    # test_default_remove_ocaml()
    # test_default_remove_query_ocaml()
    # test_default_transform_ocaml()
    # test_default_transform_query_ocaml()
    
    print(".....finished running all tests!")


# TODO: must try for better error handling in OCaml.

# TODO: test existing primitives, then switch to actual Object types in CLEVR.