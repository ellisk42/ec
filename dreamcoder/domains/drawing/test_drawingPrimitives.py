import dreamcoder.domains.drawing.drawingPrimitives as to_test
"""
Tests for makeClevrTasks.py | Author: Catherine Wong
Tests for making and loading drawing tasks.

All tests are manually added to a 'test_all' function.
"""
from types import SimpleNamespace as MockArgs

def test_load_initial_grammar_logo_primitives(args):
    mock_args = MockArgs(primitives=[to_test.LOGO_PRIMITIVES_TAG])
    mock_args = vars(mock_args)
    import dreamcoder.domains.logo.logoPrimitives as logoPrimitives
    grammar = to_test.load_initial_grammar(mock_args)
    assert len(grammar.primitives) == len(logoPrimitives.primitives)

def run_test(test_fn, args):
    """Utility function for running tests"""
    print(f"Running {test_fn.__name__}...")
    test_fn(args)
    print("\n")
    
def test_all(args):
    print("Running tests for drawingPrimitives....")
    run_test(test_load_initial_grammar_logo_primitives, args)
    print("....all tests passed!\n")