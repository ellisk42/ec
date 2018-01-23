from program import *
from type import *

class Grammar(object):
    def __init__(self, logVariable, productions):
        self.logVariable = logVariable
        self.productions = productions

    @staticmethod
    def fromProductions(productions, logVariable=0.0):
        """Make a grammar from primitives and their relative logpriors."""
        return Grammar(logVariable, [(l, p.infer(), p) for l, p in productions])

    @staticmethod
    def uniform(primitives):
        return Grammar(0.0, [(0.0,p.infer(),p) for p in primitives ])

    def __len__(self): return len(self.productions)
    def __str__(self):
        def productionKey((l,t,p)):
            return not isinstance(p,Primitive), -l
        lines = ["%f\tt0\t$_"%self.logVariable]
        for l,t,p in sorted(self.productions, key = productionKey):
            l = "%f\t%s\t%s"%(l,t,p)
            if not t.isArrow() and isinstance(p,Invented):
                l += "\teval = %s"%(p.evaluate([]))
            lines.append(l)
        return "\n".join(lines)

    @property
    def primitives(self):
        return [p for _, _, p in self.productions]

    def KL(this, that):
        assert len(this.productions) == len(that.productions) # and they should correspond
        this_z = lse([l for l, _, _ in this.productions]+[this.logVariable])
        that_z = lse([l for l, _, _ in that.productions]+[that.logVariable])

        this_l, that_l = this.logVariable, that.logVariable
        kl = exp(this_l - this_z) * (this_l - this_z - that_l + that_z)
        for i, (this_l, _, _) in enumerate(this.productions):
             that_l = that.productions[i]
             kl += exp(this_l - this_z) * (this_l - this_z - that_l + that_z)
        return kl

    @staticmethod
    def TorchKL(this_logVariable, this_productions, that):
        assert len(this_productions) == len(that.productions) # and they should correspond
        this_z = lse([lse(this_productions), this_logVariable])
        that_z = lse([l for l, _, _ in that.productions]+[that.logVariable])

        this_l, that_l = this_logVariable, that.logVariable
        kl = exp(this_l - this_z) * (this_l - this_z - that_l + that_z)
        for i, this_l in enumerate(this_productions):
             that_l, _, _ = that.productions[i]
             kl += exp(this_l - this_z) * (this_l - this_z - that_l + that_z)
        return kl


class Uses(object):
    '''Tracks uses of different grammar productions'''
    def __init__(self, possibleVariables = 0., actualVariables = 0.,
                 possibleUses = {}, actualUses = {}):
        self.actualVariables = actualVariables
        self.possibleVariables = possibleVariables
        self.possibleUses = possibleUses
        self.actualUses = actualUses

    def __str__(self):
        return "Uses(actualVariables = %f, possibleVariables = %f, actualUses = %s, possibleUses = %s)"%\
            (self.actualVariables, self.possibleVariables, self.actualUses, self.possibleUses)
    def __repr__(self): return str(self)

    def __mul__(self,a):
        return Uses(a*self.possibleVariables,
                    a*self.actualVariables,
                    {p: a*u for p,u in self.possibleUses.iteritems() },
                    {p: a*u for p,u in self.actualUses.iteritems() })
    def __rmul__(self,a):
        return self*a
    def __radd__(self,o):
        if o == 0: return self
        return self + o
    def __add__(self,o):
        if o == 0: return self
        def merge(x,y):
            z = x.copy()
            for k,v in y.iteritems():
                z[k] = v + x.get(k,0.)
            return z
        return Uses(self.possibleVariables + o.possibleVariables,
                    self.actualVariables + o.actualVariables,
                    merge(self.possibleUses,o.possibleUses),
                    merge(self.actualUses,o.actualUses))
    def __iadd__(self,o):
        self.possibleVariables += o.possibleVariables
        self.actualVariables += o.actualVariables
        for k,v in o.possibleUses:
            self.possibleUses[k] = self.possibleUses.get(k,0.) + v
        for k,v in o.actualUses:
            self.actualUses[k] = self.actualUses.get(k,0.) + v
        return self
    
Uses.empty = Uses()
