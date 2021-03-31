"""
test_configlib.py | Author : Catherine Wong
Tests for dreamcoder/configlib.py -- Configuration utilities.
"""
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

import dreamcoder.configlib as configlib

def run_test(test_fn):
    """Utility function for running tests"""
    print(f"Running {test_fn.__name__}...")
    test_fn()
    print("\n")
    
def test_abbreviate():
    """Tests that we can abbreviate and get the abbreviations of parameters."""
    TEST_VAR, TEST_ABBREVIATION = 'test_var', 'tv'
    configlib.update_config({TEST_VAR : True})
    configlib.add_abbreviation(TEST_VAR, TEST_ABBREVIATION)
    assert configlib.abbreviate(TEST_VAR) == TEST_ABBREVIATION
    assert configlib.parameterOfAbbreviation(TEST_ABBREVIATION) == TEST_VAR

def test_add_run_verifier():
    """Test that we can add and run a verifier for a given argument."""
    TEST_VAR = 'test_var'
    def verifier_true_fn(var):
        assert var
        
    configlib.update_config({TEST_VAR : True})
    configlib.add_verifier()

def test_add_run_verifier_no_arg():
    """Test that we catch verifiers for which we have no corresponding argument."""
    pass

if __name__ == '__main__':
    print("Running tests for configlib.py...")
    run_test(test_abbreviate)
    print("All tests passed!")