"""
configlib.py | Author : Catherine Wong

Contains utility functions for managing the configuration files and arguments of DreamCoder-framework experiments.

TODO:       
    Debug mode: you should be able to enable debug mode when you need to explicitly determine whether you need certain configurations.
    Verify - you should be able to define a verifier for an argument.
    Log -- we should be able to log out the argument and read arguments from a file.
    Separated: you should be able to define separate command line arguments and collect them.
    Testing -- you should be able to run tests with specific entrypoints.
    Overriding -- you should be able to define simple defaults for most experiments.
"""
from typing import Dict, Any
import logging
import pprint
import sys
import argparse

# Global parser that collects all arguments.
GLOBAL_PARSER = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")
# Global configuration dictionary that contains and accesses all parsed arguments.
config: Dict[str, Any] = {}
# Global configuration dictionary that contains abbreviations for all of the configuration parameters.
config_abbreviations = {}
# Global configuration verifier dictionary that contains all argument verifiers.
config_verifier = {}

def add_parser(title: str, description: str = ""):
    """Create a new context for arguments and return a handle."""
    return GLOBAL_PARSER.add_argument_group(title, description)

def save_arguments(save_fname: str = ""):
    """Optionally saves passed arguments."""
    if save_fname:
        with open(save_fname, "w") as fout:
            fout.write("\n".join(sys.argv[1:]))

def parse(save_fname: str = "") -> Dict[str, Any]:
    """Parse given arguments"""
    config.update(vars(GLOBAL_PARSER.parse_args()))
    save_arguments(save_fname)
    return config

def update_config(manual_config_vars, save_fname: str = ""):
    """Manually updates the global configuration dictionary with any argumetns."""
    config.update(manual_config_vars)
    # Optionally save passed arguments
    save_arguments(save_fname)
    return config

### Abbreviations for parameter names.
def add_abbreviation(arg, abbreviation):
    """Adds a short abbreviation for the parameter (e.g. for constructing checkpoints). Ensures that the abbreviation is also unique."""
    if abbreviation in parameters_to_abbreviations():
        original_arg = parameters_to_abbreviations()[abbreviation]
        print(f"Abbreviation {abbreviation} already exists for arg: {original_arg}")
        assert False
    if arg in config_abbreviations:
        print(f"Arg {arg} already exists with abbreviation: {abbreviation}")
        assert False
    else:
        config_abbreviations[arg] = abbreviation

def abbreviate(arg):
    """Returns the abbreviated name of the parameter or the original name if there is no abbreviation"""
    return config_abbreviations.get(arg, arg)

def abbreviate_value(value):
    """Returns an abbreviated string of the value if there is one or the original value"""
    if type(value) == bool:
        return str(value)[0]
    else:
        return value

def parameters_to_abbreviations():
    """Convenience method to invert the config->abbreviation dictionary."""
    return {v : k for k, v in config_abbreviations.items()}
    
def parameterOfAbbreviation(abbreviation):
    """Restores the original parameter from the abbreviation."""
    return parameters_to_abbreviations().get(abbreviation, abbreviation)

### Verifications for parameters.
def add_verifier(arg, verifier_fn):
    """Adds or updates a verification function for a given argument."""
    pass

def verify_not_none(arg, msg=None):
    """Automatically verifies that a given argument is not None."""
    
def verify_arg(arg, verifier_fn):
    """Runs a verification for a singular argument."""
    pass

def verify_all():
    """Adds a verification function"""
    pass


def print_config():
    """Print the current config to stdout."""
    pprint.pprint(config)
    
    