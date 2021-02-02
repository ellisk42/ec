"""
cogsci_utils | Author : Catherine Wong.
Utility functions for generating conditions for the human experiments.
"""
LANGUAGE_CONDITION = 'language'
NON_LANGUAGE_CONDITION = 'no-language'
ENUMERATION_TIMEOUT = 1000
RECOGNITION_STEPS = 10000

def get_shared_experiment_parameters():
    # batch size
    # number of iterations : 3
    # test on every: 1
    # enumeration_timeout 
    pass
    

def load_language_condition_arguments():
    f"--recognition_0 --recognition_1 examples language --Helmholtz 0.5 --primitives {primitives_string}  --synchronous_grammar "
    f"--lc_score 0 "
    f"--smt_pseudoalignments {pseudoalignments_weight}"

def load_no_language_condition_arguments():
    " --recognition_0 examples --Helmholtz 0.5 --primitives clevr_bootstrap clevr_map_transform "
    

def load_condition_args(args):
    """
    Generates the complete set of running information necessary to replicate the language vs. non-language runs.
    Returns {
        'language' : language_condition_dict; 'no-language': no-language_condition_dict
    }
    """
    return
    