"""
Tests for clevrRecognition.py | Author: Catherine Wong
Test for the neural recognition model specific to the CLEVR symbolic scene graph dataset.

All tests are manually added to a 'test_all' function.

The tests are based on the recognition model(s) in clevrRecognition.py.
"""

def test_recognition_model():
    pass


def run_test(test_fn):
    """Utility function for running tests"""
    print(f"Running {test_fn.__name__}...")
    test_fn()
    print("\n")

def test_all():
    print("Running tests for clevrRecognition...")
    run_test(test_recognition_model)
    print(".....finished running all tests!")