#RobustFillPrimitives

from dreamcoder.program import Primitive, Program, prettyProgram
from dreamcoder.grammar import Grammar
from dreamcoder.type import tlist, tint, tbool, arrow, baseType #, t0, t1, t2

import math
from string import printable
import re
from collections import defaultdict
from dreamcoder.utilities import timing
#from functools import reduce

import dreamcoder.ROB as ROB

from dreamcoder.ROB import allowed


delimiters = "&,.?!@()[]%{/}:;$#\"' "

delim_dict = {disallowed[c]:c for c in delimiters}

tposition = baseType("position")
tindex = baseType("index")
tcharacter = baseType("character")
tboundary = baseType("boundary")
tregex = baseType("regex")
tsubstr = baseType("substr")
texpression = baseType("expression")
tprogram = baseType("program")
tnesting = baseType("nesting")
ttype = baseType("type")
tdelimiter = baseType("delimiter")


def _getSpan(r1): 
    return lambda i1: lambda b1: lambda r2: lambda i2: lambda b2: lambda k: [ROB.GetSpan(r1, i1, b1, r2, i2, b2)] + k

def robustFillPrimitives():

    return [
        #substring:
        CPrimitive("SubStr", arrow(tposition, tposition, texpression, texpression), lambda i: lambda j: lambda k: [ROB.SubString(i, j)] + k),
        CPrimitive("GetSpan", arrow(tregex, tindex, tboundary, tregex, tindex, tboundary, texpression, texpression), _getSpan ),
        #nesting:
        CPrimitive("GetToken", arrow(ttype, tindex, texpression, texpression), lambda t: lambda i: lambda k: [ROB.GetToken(t, i) ] + k),   
        CPrimitive("ToCase_Proper", arrow(texpression, texpression), lambda k: [ROB.ToCase( ("Proper", lambda x : x.title()) )] + k),
        CPrimitive("ToCase_AllCaps", arrow(texpression, texpression), lambda k: [ROB.ToCase( ("AllCaps", lambda x: x.upper()) )] + k),
        CPrimitive("ToCase_Lower", arrow(texpression, texpression), lambda k: [ROB.ToCase( ("Lower", lambda x: x.lower()) )] + k),
        CPrimitive("Replace", arrow(tdelimiter, tdelimiter, texpression, texpression),  lambda d1: lambda d2: lambda k: [ROB.Replace(d1, d2)] + k ), 
        CPrimitive("GetUpTo", arrow(tregex, texpression, texpression), lambda r: lambda k: [ROB.GetUpTo(r)] + k ),
        CPrimitive("GetFrom", arrow(tregex, texpression, texpression),  lambda r: lambda k: [ROB.GetFrom(r)] + k ),
        CPrimitive("GetFirst", arrow(ttype, tindex, texpression, texpression),  lambda t: lambda i: lambda k: [ROB.GetFirst(t, i)] + k ) ,
        CPrimitive("GetAll", arrow(ttype, texpression, texpression), lambda r: lambda k: [ROB.GetAll(r)] + k ),
        #n versions
        CPrimitive("GetToken_n", arrow(texpression, ttype, tindex, texpression, texpression), lambda e: lambda t: lambda i: lambda k: [ROB.Compose(ROB.GetToken(t, i), e) ] + k),   
        CPrimitive("ToCase_Proper_n", arrow(texpression, texpression, texpression), lambda e: lambda k: [ROB.Compose(ROB.ToCase( ("Proper", lambda x : x.title()) ), e)] + k),
        CPrimitive("ToCase_AllCaps_n", arrow(texpression, texpression, texpression), lambda e: lambda k: [ROB.Compose(ROB.ToCase( ("AllCaps", lambda x: x.upper()) ), e)] + k),
        CPrimitive("ToCase_Lower_n", arrow(texpression, texpression, texpression), lambda e: lambda k: [ROB.Compose(ROB.ToCase( ("Lower", lambda x: x.lower()) ), e)] + k),
        CPrimitive("Replace_n", arrow(texpression, tdelimiter, tdelimiter, texpression, texpression),  lambda e: lambda d1: lambda d2: lambda k: [ROB.Compose(ROB.Replace(d1, d2), e)] + k ), #TODO
        CPrimitive("GetUpTo_n", arrow(texpression, tregex, texpression, texpression), lambda e: lambda r: lambda k: [ROB.Compose(ROB.GetUpTo(r), e)] + k ),
        CPrimitive("GetFrom_n", arrow(texpression, tregex, texpression, texpression),  lambda e: lambda r: lambda k: [ROB.Compose(ROB.GetFrom(r), e)] + k ),
        CPrimitive("GetFirst_n", arrow(texpression, ttype, tindex, texpression, texpression),  lambda e: lambda t: lambda i: lambda k: [ROB.Compose(ROB.GetFirst(t, i),e)] + k ) ,
        CPrimitive("GetAll_n", arrow(texpression, ttype, texpression, texpression), lambda e: lambda r: lambda k: [ROB.Compose(ROB.GetAll(r), e)] + k ),
        ] + [
        #Regex
        CPrimitive(f"regex_{allowed(r)}", tregex, r) for r in ROB._POSSIBLE_R.keys()
        #type
        ] + [
        CPrimitive(f"type_{tp}", ttype, tp) for tp in ROB._POSSIBLE_TYPES.keys()
        ] + [
        #position
        CPrimitive(f"pos_{i}", tposition, i) for i in ROB._POSITION_K
        ] + [
        #index
        CPrimitive(f"index_{i}", tposition, i) for i in ROB._INDEX
        ] + [
        #Character
        CPrimitive(f"char_{allowed(c)}", texpression, texpression, lambda k: [ROB.ConstStr(c)] + k) for c in ROB._CHARACTER
        ] + [
        #delimiter
        CPrimitive(f"delim_{allowed(d)}", tdelimiter, d) for d in ROB._DELIMITER
        ] + [
        #boundary
        CPrimitive(f"bound_{b}", tboundary, b) for b in ROB._BOUNDARY
        ]


def RobustFillProductions(max_len=100, max_index=5):
    return [(0.0, prim) for prim in robustFillPrimitives(max_len=max_len, max_index=max_index)]


def flatten_program(p):
    string = p.show(False)
    string = string.replace('(', '')
    string = string.replace(')', '')
    #remove '_fn' (optional)
    string = string.split(' ')
    string = list(filter(lambda x: x is not '', string))
    return string




def add_constraints(c1, c2=None):
    if c2 is None:
        return c1
    d1, m1 = c1
    d2, m2 = c2
    min_size = max(m1, m2)
    d = defaultdict(int)
    for item in set(d1.keys()) | set(d2.keys()):
        d[item] = max(d1[item], d2[item])
    return d, min_size

# class Constraint_prop:
#     def application(self, p, environment):
#         self.f.visit(self, environment)(self.x.visit(self, environment))
#     def primitive(self, p, environment):
#         return self.value

class Constraint_prop:
    def __init__(self):
        pass

    def application(self, p):
        return p.f.visit(self)(p.x.visit(self))

    def primitive(self, p):
        return p.constraint

    def execute(self, p):
        return p.visit(self)
    

class CPrimitive(Primitive):
    def __init__(self, name, ty, value, constraint=None):
        #I have no idea why this works but it does ..... 
        if constraint is None:
            if len(ty.functionArguments())==0:
                self.constraint = (defaultdict(int), 0)
            elif len(ty.functionArguments())==1:
                self.constraint = lambda x: x
            elif len(ty.functionArguments())==2:
                self.constraint = lambda x: lambda y: add_constraints(x,y)
            else:
                self.constraint = lambda x: x
                for _ in range(len(ty.functionArguments()) - 1):
                    self.constraint = lambda x: lambda y: add_constraints(x, self.constraint(y))
        else: self.constraint = constraint
        super(CPrimitive, self).__init__(name, ty, value)

    #def __getinitargs__(self):
    #    return (self.name, self.tp, self.value, None)

    def __getstate__(self):
        #print("self.name", self.name)
        return self.name

    def __setstate__(self, state):
        #for backwards compatibility:
        if type(state) == dict:
            pass #do nothing, i don't need to load them if they are old...
        else:
            p = Primitive.GLOBALS[state]
            self.__init__(p.name, p.tp, p.value, p.constraint) 



if __name__=='__main__':
    import time
    CPrimitive("testCPrim", tint, lambda x: x, 17)
    g = Grammar.fromProductions(RobustFillProductions())
    print(len(g))
    request = tprogram
    p = g.sample(request)
    print("request:", request)
    print("program:")
    print(prettyProgram(p))
    s = 'abcdefg'
    e = p.evaluate([])
    #print("prog applied to", s)
    #print(e(s))
    print("flattened_program:")
    flat = flatten_program(p)
    print(flat)
    t = time.time()    
    constraints = Constraint_prop().execute(p)
    print(time.time() - t)
    print(constraints)