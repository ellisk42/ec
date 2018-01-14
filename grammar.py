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
        lines = ["%f\tt0\t$_"%self.logVariable]
        for l,t,p in self.productions:
            l = "%f\t%s\t%s"%(l,t,p)
            if not t.isArrow() and isinstance(p,Invented):
                l += "\teval = %s"%(p.evaluate([]))
            lines.append(l)
        return "\n".join(lines)
