"""
laps.py | Author : Catherine Wong

Framework for hierarchical Bayesian models that synthesize and induce over programs and language.
Interfaces modularly with the codebase in DreamCoder.
"""

import dreamcoder.checkpoint as checkpoint
import dreamcoder.configlib as configlib

laps_parser = configlib.add_parser("laps.py")
laps_parser.add_argument("--laps", action="store_true", help="Enable LAPS experimental framework instead of DreamCoder.")


# TODO: implement a result object.

# TODO: implement a global dictionary for the experiment segments.
# You should be able to just run a model configuration at each step.
experiment_block_registry = {}


def LAPSIterator(*args, **kwargs):
    initialize_state_and_tasks()
    
    # Get the next task batch for the number of iterations.
    # Get the next experiment segment
    for experiment_block in experiment_blocks:
        pass
        # Update the joint-model: Fit P(programs, descriptions | library)
        
        # Search from the prior: sample ~ P(programs, descriptions | library)
        
        # Update and search from the conditional model: P(programs | descriptions, examples, library)
        
        # Update the program prior: Fit P(library | discovered programs)
    
    # Store a checkpoint.

def initialize_state_and_tasks():
    """Initializes or resumes an experiment state, including the tasks that it was run on.
    Returns:
        Initialized LAPSExperimentState.
    """
    pass
    # Verify configuration arguments.
    
    # Log all configuration parameters.
    
    # Load the language dataset if available.
    
    # Initialize or resume the full model checkpoint state.
    pass

def evaluate_heldout_tasks():
    """Evaluates on heldout tasks and stores the result."""
    pass


    
    
    
    