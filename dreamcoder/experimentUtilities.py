"""
experimentUtilities.py | Author: Catherine Wong

Utilities for running scheduled iterative experiments.
"""
from dreamcoder.dreamcoder import ecIterator
from dreamcoder.laps import LAPSIterator
from dreamcoder.utilities import pop_all_domain_specific_args

EXPERIMENT_ITERATORS = {
    'laps' : LAPSIterator
}

def maybe_pop_domain_specific_arguments(iterator_type):
    """Utility to pop off domain-specific arguments. Used for backwards compatability with iterators with a set number of arguments."""
    if iterator_type == ecIterator:
        pop_all_domain_specific_args(args_dict=args, iterator_fn=experiment_iterator)
    
def run_scheduled_iterative_experiment(args, domain_specific_args, train_test_schedules):
    """
    Runs iterative experiments with a train and test schedule of tasks.
    Runs an indicated iterator, or uses DreamCoder experiments as a fallback.
    """
    for schedule_idx, (train_tasks, test_tasks) in enumerate(train_test_schedules):
        print(f"Train-test schedules: [{schedule_idx}/{len(train_test_schedules)}]. Using {len(train_tasks)} train / {len(test_tasks)} test tasks.")
        domain_specific_args["tasks"] = train_tasks
        domain_specific_args["testingTasks"] = test_tasks
        

        experiment_iterator = ecIterator # Default is DreamCoder
        for iterator_arg in EXPERIMENT_ITERATORS:
            if iterator_arg in args:
                experiment_iterator =  EXPERIMENT_ITERATORS[iterator_arg]
        
        # Utility to pop off any additional arguments that are specific to this domain.
        maybe_pop_domain_specific_arguments(experiment_iterator)
        generator = experiment_iterator(**domain_specific_args,
                               **args)
        for result in generator:
            pass
    
    