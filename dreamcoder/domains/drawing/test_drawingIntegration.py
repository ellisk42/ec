"""
Integration tests for drawing/main.py | Author : Catherine Wong.

These are full integration tests for the functionality of the various components
of a DreamCoder iteration using the drawing datasets.

All tests are manually added to a 'test_all' function.

Usage: python bin/clevr.py --run_drawingIntegration_test
"""
from dreamcoder.utilities import *
from dreamcoder.dreamcoder import ecIterator

def set_default_args(args):
    """Helper function to set default arguments in the args dictionary."""
    args['contextual'] = True
    args['biasOptimal'] = True
    args['taskBatchSize'] = 10
    args['enumerationTimeout'] = 1
    
def run_test(test_fn, DOMAIN_SPECIFIC_ARGS, args, extras=None):
    """Utility function for running tests"""
    print(f"Running {test_fn.__name__}...")
    test_fn(DOMAIN_SPECIFIC_ARGS, args, extras)
    print("\n")

def test_drawing_specific_command_line_args(DOMAIN_SPECIFIC_ARGS, args, extras):
    """
    Test that we correctly use and remove all drawing-specific command line arguments.
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
    assert DOMAIN_SPECIFIC_ARGS["tasks"] is None # We run on a schedule.
    assert DOMAIN_SPECIFIC_ARGS["testingTasks"] is None 
    assert len(DOMAIN_SPECIFIC_ARGS["grammar"]) > 0
    # Check that the output prefix was created.
    checkpoint_dir =os.path.dirname(DOMAIN_SPECIFIC_ARGS["outputPrefix"])
    assert os.path.isdir(checkpoint_dir)
    assert args["iterations"] > 1

def test_train_test_schedules(DOMAIN_SPECIFIC_ARGS, args, train_test_schedules):
    """Test that we can correctly load tasks into the iterator on a schedule. This contains a copy of logic in main/drawing.py"""
    assert len(train_test_schedules) > 0
    set_default_args(args)
    for schedule_idx, (train_tasks, test_tasks) in enumerate(train_test_schedules):
        DOMAIN_SPECIFIC_ARGS["tasks"] = train_tasks
        DOMAIN_SPECIFIC_ARGS["testingTasks"] = test_tasks
        generator = ecIterator(**DOMAIN_SPECIFIC_ARGS,
                               **args,
                               test_tasks=True)
        for current_ec_result in generator:
            num_test_tasks = current_ec_result.numTestingTasks
            num_train_tasks = len(current_ec_result.allFrontiers)
            break
        
        assert num_test_tasks == len(set(test_tasks))
        assert num_train_tasks == len(set(train_tasks))

def test_integration_task_language_synthetic(DOMAIN_SPECIFIC_ARGS, args, train_test_schedules):
    """Test that we correctly load all synthetic language for a schedule of LOGO tasks"""
    assert len(train_test_schedules) > 0
    set_default_args(args)
    for schedule_idx, (train_tasks, test_tasks) in enumerate(train_test_schedules):
        DOMAIN_SPECIFIC_ARGS["tasks"] = train_tasks
        DOMAIN_SPECIFIC_ARGS["testingTasks"] = test_tasks
        generator = ecIterator(**DOMAIN_SPECIFIC_ARGS,
                               **args,
                               test_task_language=True)
        for current_ec_result in generator:
            language_for_tasks, vocabularies = current_ec_result.taskLanguage, current_ec_result.vocabularies
            break
            
        assert len(vocabularies['train']) > 0
        assert len(vocabularies['test']) > 0
        
        # Check that all of the tasks have language.
        for task in DOMAIN_SPECIFIC_ARGS['tasks']:
            assert task.name in language_for_tasks
            assert len(language_for_tasks) > 0
            
        for task in DOMAIN_SPECIFIC_ARGS['testingTasks']:
            assert task.name in language_for_tasks
            assert len(language_for_tasks) > 0
    

def test_all(DOMAIN_SPECIFIC_ARGS, args, train_test_schedules):
    print("Running tests for clevrIntegration.py...")
    run_test(test_drawing_specific_command_line_args, DOMAIN_SPECIFIC_ARGS, args)
    run_test(test_train_test_schedules, DOMAIN_SPECIFIC_ARGS, args, train_test_schedules)
    run_test(test_integration_task_language_synthetic, DOMAIN_SPECIFIC_ARGS, args, train_test_schedules)