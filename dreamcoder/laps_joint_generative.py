"""
joint_generative.py | Author : Catherine Wong

Interface for joint generative models over programs and language. 
Used for fitting, sampling, and evaluating. 
Also contains definitions for common commandline arguments for joint-model-specific parameters.

Local configuration arguments:
"""
import dreamcoder.configlib as configlib
from configlib import config as C

# Local configuration arguments.
joint_parser = configlib.add_parser("joint_model")

# Joint model registry for any models.
JOINT_MODEL_REGISTRY = {}

class LAPSJointGenerative():
    """LAPSJointGenerative: A joint generative model over language and programs.
    """
    def __init__(self):
        pass
    
    def fit(self):
        """Fits the joint generative model. Generally expects to be fit to 
        tasks with accompanying natural language.
        """
        # Find the appropriate model and call fit.
        pass
    
    def sample_joint(self):
        """Samples from the joint generative model.
        Returns: Task with accompanying language.
        """
        pass
    
    def evaluate(self):
        """Evaluates the joint generative model."""
        pass
# Register the name into the registry.

class LAPSProgramPriorTranslationJoint(LAPSJointGenerative):
    """LAPSProgramPriorTranslationJoint: A common pattern for a joint generative model 
    that decomposes as a program prior and a program->language translation model."""
    def __init__(self):
        pass
    
    def fit(self):
        """Fits the joint generative model. Fits the 
        """
        pass
        
    def sample_program_prior(self):
        pass
    
    def sample_language_for_program(self):
        pass
    
    def sample_joint(self):
        """Samples from the joint generative model.
        Samples first from the program prior, then samples from the program-language model.
        Returns: Task with accompanying language.
        """
        pass
    
    def evaluate(self):
        """Evaluates the joint generative model."""
        pass


    

