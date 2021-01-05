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

def check_ocaml_evaluation_for_task(task, programs_to_test):
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
    assert task.name in response
    did_succeed =  response[task.name]
    assert did_succeed

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
    
    print(".....finished running all tests!")


# TODO: must try for better error handling in OCaml.

# TODO: test existing primitives, then switch to actual Object types in CLEVR.