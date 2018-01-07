from program import *
from type import *


class Grammar(object):
    def __init__(self, logVariable, productions):
        self.logVariable = logVariable
        self.productions = productions

    @staticmethod
    def uniform(primitives):
        return Grammar(0.0, [(0.0,p.infer(),p) for p in primitives ])
