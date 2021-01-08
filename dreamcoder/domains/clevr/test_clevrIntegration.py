"""
Integration tests for clevr/main.py | Author : Catherine Wong.

These are full integration tests for the functionality of the various components
of a DreamCoder iteration using the CLEVR dataset. 

All tests are manually added to a 'test_all' function.
"""
from dreamcoder.utilities import DEFAULT_OUTPUT_DIRECTORY, pop_all_domain_specific_args
from dreamcoder.dreamcoder import ecIterator
import inspect
import os
def test_clevr_specific_command_line_args(DOMAIN_SPECIFIC_ARGS, args):
    """
    Test that we correctly use and remove all CLEVR-specific command line arguments.
    """
    # We have to pop off all domain-specific arguments here, or we'll not have the 
    # argument to actually run the intergration test.
    pop_all_domain_specific_args(args_dict=args, iterator_fn=ecIterator)
    # Check that we've removed any arguments that aren't present in the ecIterator.
    ecIterator_parameters = inspect.signature(ecIterator).parameters
    for arg in DOMAIN_SPECIFIC_ARGS:
        assert arg in ecIterator_parameters
    for arg in args:
        assert arg in ecIterator_parameters
    # Integration checks on the actual parameters.
    assert len(DOMAIN_SPECIFIC_ARGS["tasks"]) > 0
    assert len(DOMAIN_SPECIFIC_ARGS["testingTasks"]) > 0
    assert "1_zero_hop" in DOMAIN_SPECIFIC_ARGS["languageDataset"]
    assert "2_transform" in DOMAIN_SPECIFIC_ARGS["languageDataset"]
    assert len(DOMAIN_SPECIFIC_ARGS["grammar"]) > 0
    # Check that the output prefix was created.
    checkpoint_dir =os.path.dirname(DOMAIN_SPECIFIC_ARGS["outputPrefix"])
    assert os.path.isdir(checkpoint_dir)
    assert args["iterations"] > 1

def test_integration_task_language(DOMAIN_SPECIFIC_ARGS, args):
    pop_all_domain_specific_args(args_dict=args, iterator_fn=ecIterator)
    args['enumerationTimeout'] = 0
    generator = ecIterator(**DOMAIN_SPECIFIC_ARGS, **args,
     test_task_language=True)
    
    for current_ec_result in generator:
        language_for_tasks, vocabularies = current_ec_result.taskLanguage, current_ec_result.vocabularies
        break
        
    assert len(vocabularies['train']) > 0
    assert len(vocabularies['test']) > 0
    
    # Check that all of the tasks ave language.
    for task in DOMAIN_SPECIFIC_ARGS['tasks']:
        assert task.name in language_for_tasks
        assert len(language_for_tasks) > 0
        
    for task in DOMAIN_SPECIFIC_ARGS['testingTasks']:
        assert task.name in language_for_tasks
        assert len(language_for_tasks) > 0

def run_test(test_fn, DOMAIN_SPECIFIC_ARGS, args):
    """Utility function for running tests"""
    print(f"Running {test_fn.__name__}...")
    test_fn(DOMAIN_SPECIFIC_ARGS, args)
    print("\n")

def test_all(DOMAIN_SPECIFIC_ARGS, args):
    print("Running tests for clevrIntegration.py...")
    # run_test(test_clevr_specific_command_line_args, DOMAIN_SPECIFIC_ARGS, args)
    run_test(test_integration_task_language, DOMAIN_SPECIFIC_ARGS, args)
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

