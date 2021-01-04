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

def check_ocaml_evaluation_for_task(task, raw_program_original, raw_program_bootstrap):
    TIMEOUT = 10
    # Construct a grammar object.
    clevr_primitives = clevrPrimitives.load_clevr_primitives(CLEVR_PRIMITIVE_SETS)
    clevr_grammar = Grammar.uniform(clevr_primitives)
    
    # Add the desired programs to the task to be tested.
    task.raw_programs_to_test = [raw_program_original, raw_program_bootstrap]
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
    check_ocaml_evaluation_for_task(task, raw_program_original, raw_program_bootstrap)
    print("\n")

def test_all():
    print("Running tests for clevrPrimitivesOcaml....")
    test_localization_task_ocaml()
    
    print(".....finished running all tests!")

# TODO: see enumeration.py for the functionality to call a single OCaml task.

# TODO: see CLEVR test -- may be able to substitute in a special task.specialSolver to use the test functions, and pass in the raw program directly.

# TODO: must try for better error handling in OCaml.

# TODO: test existing primitives, then switch to actual Object types in CLEVR.