"""
simple_joint.py | Author : Catherine Wong

Defines simple joint generative models.
"""

class LAPSProgramPriorOnlyJoint(LAPSJointGenerative):
    """LAPSProgramPriorOnlyJoint: A baseline degenerate model that only samples 
    from the program prior. Using this is equivalent to training on DreamCoder. 
     """
    pass

class LAPS_IBMTranslationJoint(LAPSProgramPriorTranslationJoint):
    """LAPS_IBMTranslationJoint: A simple program-prior + translation-model joint 
    model that uses the IBM model as its program->language translation model.
    This re-implements the model used in the Wong et. al 2020 paper.
    """
    pass
