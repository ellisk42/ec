import dreamcoder.domains.logo.logoPrimitives as logoPrimitives
from dreamcoder.grammar import Grammar

"""
drawingPrimitives.py | Author: Catherine Wong

Loads primitives designed to produce structured drawings.
This contains a reimplementation based on the LOGO Graphics language used in DreamCoder (Ellis et. al 2020); and the graphics language used in Tian et. al 2020.
"""
LOGO_PRIMITIVES_TAG = "logo"
TIAN_PRIMITIVES_TAG = "tian"

def load_initial_grammar(args):
    """
    Loads primitives and returns an initialized Grammar.
    """
    primitive_classes = args['primitives']
    loaded_primitives = []
    for primitive_class_name in primitive_classes:
        if LOGO_PRIMITIVES_TAG in primitive_class_name :
            primitives = logoPrimitives.primitives
            return Grammar.uniform(primitives, continuationType=logoPrimitives.turtle)
        elif TIAN_PRIMITIVES_TAG in primitive_class_name :
            print(f"Not yet implemented: {primitive_name}")
            assert False
        else:
            print(f"Not found - primitives for: {primitive_class_name }")
            assert False
    