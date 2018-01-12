from program import *
from type import *


class Grammar(object):
    def __init__(self, logVariable, productions):
        self.logVariable = logVariable
        self.productions = productions

    @staticmethod
    def uniform(primitives):
        return Grammar(0.0, [(0.0,p.infer(),p) for p in primitives ])

    def __len__(self): return len(self.productions)
    def __str__(self):
        return "\n".join(["%f\tt0\t$_"%self.logVariable] + [ "%f\t%s\t%s"%(l,t,p) for l,t,p in self.productions ])
