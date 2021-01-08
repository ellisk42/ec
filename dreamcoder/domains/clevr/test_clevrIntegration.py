"""
Integration tests for clevr/main.py | Author : Catherine Wong.

These are full integration tests for the functionality of the various components
of a DreamCoder iteration using the CLEVR dataset. 

All tests are manually added to a 'test_all' function.
"""
def test_clevr_specific_command_line_args():
    """
    Test that we correctly use and remove all CLEVR-specific command line arguments.
    """
    pass


def run_test(test_fn):
    """Utility function for running tests"""
    print(f"Running {test_fn.__name__}...")
    test_fn()
    print("\n")

def test_all():
    print("Running tests for clevrIntegration.py...")
    run_test(test_recognition_model)
    print(".....finished running all tests!")
    


# if args.pop("run_ocaml_test"):
#     # Test the Helmholtz enumeration
#     # tasks = [buildClevrMockTask(train[0])]
#     tasks = train[:10]
#     if True:
#         from dreamcoder.dreaming import backgroundHelmholtzEnumeration
#         print(baseGrammar)
#         helmholtzFrontiers = backgroundHelmholtzEnumeration(tasks, 
#                                                             baseGrammar, 
#                                                             timeout=5,
#                                                             evaluationTimeout=0.05,
#                                                             special='clevr',
#                                                             executable='clevrTest',
#                                                             serialize_special=serialize_clevr_object,
#                                                             maximum_size=20000) # TODO: check if we need special to check tasks later
#         f = helmholtzFrontiers()
#         helmholtzFrontiers = backgroundHelmholtzEnumeration(train, 
#                                                             baseGrammar, 
#                                                             timeout=5,
#                                                             evaluationTimeout=0.05,
#                                                             special='clevr',
#                                                             executable='helmholtz',
#                                                             serialize_special=serialize_clevr_object,
#                                                             maximum_size=20000) # TODO: check if we need special to check tasks later
#         f = helmholtzFrontiers()
#         assert False
#     if False:
#         # Check enumeration.
#         tasks = [train[10]]
#         default_wake_generative(baseGrammar, tasks, 
#                             maximumFrontier=5,
#                             enumerationTimeout=1,
#                             CPUs=1,
#                             solver='ocaml',
#                             evaluationTimeout=0.05)
#         assert False

