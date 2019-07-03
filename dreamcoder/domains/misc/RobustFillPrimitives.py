#RobustFillPrimitives

from dreamcoder.program import Primitive, prettyProgram
from dreamcoder.grammar import Grammar
from dreamcoder.type import tint, arrow, baseType #, t0, t1, t2

from string import printable
import re
from collections import defaultdict

#from functools import reduce


disallowed = [
    ("#", "hash"),
    ("!", "bang"),
    ("\"", "double_quote"),
    ("$", "dollar"),
    ("%", "percent"),
    ("&", "ampersand"),
    ("'", "single_quote"),
    (")", "left_paren"),
    ("(", "right_paren"),
    ("*", "astrisk"),
    ("+", "plus"),
    (",", "comma"),
    ("-", "dash"),
    (".", "period"),
    ("/", "slash"),
    (":", "colon"),
    (";", "semicolon"),
    ("<", "less_than"),
    ("=", "equal"),
    (">", "greater_than"),
    ("?", "question_mark"),
    ("@", "at"),
    ("[", "left_bracket"),
    ("\\", "backslash"),
    ("]", "right_bracket"),
    ("^", "carrot"),
    ("_", "underscore"),
    ("`", "backtick"),
    ("|", "bar"),
    ("}", "right_brace"),
    ("{", "left_brace"),
    ("~", "tilde"),
    (" ", "space"),
    ("\t", "tab")
]
disallowed = dict(disallowed)
delimiters = "&,.?!@()[]%{/}:;$#\"'"

delim_dict = {disallowed[c]:c for c in delimiters}

types = {}
types["Number"] = r'\d+'
types["Word"] = r'\w+'
types["Alphanum"] = r'\w'
types["PropCase"] = r'[A-Z][a-z]+'
types["AllCaps"] = r'[A-Z]'
types["Lower"] = r'[a-z]'
types["Digit"] = r'\d'
types["Char"] = r'.' 

regexes = {name: re.escape(val) for name, val in delim_dict.items()}
regexes = {**regexes, **types}

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

def _substr(k1): return lambda k2: lambda string: string[k1:k2] #i think this is fine
def _getspan(r1): 
    return lambda i1: lambda b1: lambda r2: lambda i2: lambda b2: lambda string: \
    string[
    [m.end() for m in re.finditer(r1, string)][i1] if b1 == "End" else [m.start() for m in re.finditer(r1, string)][i1]:[m.end() for m in re.finditer(r2, string)][i2] if b2 == "End" else [m.start() for m in re.finditer(r2, string)][i2]
    ]
    #TODO format correctly
def _getspan_const(r1): return lambda i1: lambda b1: lambda r2: lambda i2: lambda b2: (defaultdict(int, {r1:i1+1 if i1>=0 else abs(i1), r2:i2+1 if i2>=0 else abs(i2)}), max(i1+1 if i1>=0 else abs(i1), i2+1 if i2>=0 else abs(i2)))


def _trim(string): 
    assert False
    return string

def _replace(d1, d2): return lambda string: string.replace(d1,d2)

def _getall(tp): return lambda string: ''.join(re.findall(tp, string))
def _getfirst(tp, i): return lambda string: ''.join(re.findall(tp, string)[:i])
def _gettoken(tp, i): return lambda string: re.findall(tp, string)[i]
def _gettoken_const(tp, i): return defaultdict(int, {tp: i+1 if i>=0 else abs(i)}), i+1 if i>=0 else abs(i)

def _getupto(reg): return lambda string: string[:[m.end() for m in re.finditer(reg, string)][0]]
def _getfrom(reg): return lambda string: string[[m.end() for m in re.finditer(reg, string)][-1]:]

def _concat2(expr1): return lambda expr2: lambda string: expr1(string) + expr2(string) #More concats plz
def _concat1(expr): return lambda string: expr(string)
def _concat_list(expr): return lambda program: lambda string: expr(string) + program(string)
#i've decided that all of the things which are expressions should take tstring as last input and output a tstring. Thus, all requests are arrow(tstring, tstring) and we limit size with recursive depth
"""
todo: 
- _trim
- incorporate tcharacter 
- constraints
- format _getspan
- figure out how to represent on top_level

- flatten for nn
- parse

- robustfill_util
- train dc model for robustfill
- main_supervised_robustfill
- evaluate_robustfill
- sample_data


- deal with escapes ... 

constraints:
elements, and number necessary, and lengths
"""

def robustFillPrimitives(max_len=100, max_index=5):
    return [
        #CPrimitive("concat2", arrow(texpression, texpression, tprogram), _concat2),
        CPrimitive("concat1", arrow(texpression, tprogram), _concat1),
        CPrimitive("concat_list", arrow(texpression, tprogram, tprogram), _concat_list),
        #expressions
        CPrimitive("Constant", arrow(tcharacter, texpression), lambda x: lambda y: x),  # add a constraint
        CPrimitive("apply", arrow(tnesting, tsubstr, texpression), lambda n: lambda sub: lambda string: n(sub(string))),
        CPrimitive("apply_n", arrow(tnesting, tnesting, texpression), lambda n1: lambda n2: lambda string: n1(n2(string))),
        CPrimitive("expr_n", arrow(tnesting, texpression), lambda x: x),
        CPrimitive("expr_f", arrow(tsubstr, texpression), lambda x: x)
        ] + [
        #substrings
        CPrimitive("SubStr", arrow(tposition, tposition, tsubstr), _substr), # handled
        CPrimitive("GetSpan", arrow(tregex, tindex, tboundary, tregex, tindex, tboundary, tsubstr), _getspan, _getspan_const)  #TODO constraint
        ] + [
        #nestings
        CPrimitive("GetToken"+name+str(i), tnesting, _gettoken(tp,i), _gettoken_const(tp, i)) for name, tp in types.items() for i in range(-max_index, max_index)
        ] + [
        CPrimitive("ToCase_ProperCase", tnesting, lambda x: x.title(), (defaultdict(int, {r'[A-Z][a-z]+':1}), 1)),
        CPrimitive("ToCase_AllCapsCase", tnesting, lambda x: x.upper(), (defaultdict(int, {r'[A-Z]':1}) ,1)),
        CPrimitive("ToCase_LowerCase", tnesting, lambda x: x.lower(), (defaultdict(int, {r'[a-z]':1}), 1) )
        ] + [
        CPrimitive("Replace_"+name1+name2, tnesting, _replace(char1, char2), (defaultdict(int, {char1:1}), 1)) for name1, char1 in delim_dict.items() for name2, char2 in delim_dict.items() if char1 is not char2
        ] + [
        #CPrimitive("Trim", tnesting, _trim), #TODO
        ] + [
        CPrimitive("GetUpTo"+name, tnesting, _getupto(reg), (defaultdict(int, {reg:1} ),1)) for name, reg in regexes.items()
        ] + [
        CPrimitive("GetFrom"+name, tnesting, _getfrom(reg), (defaultdict(int, {reg:1} ),1)) for name, reg in regexes.items()
        ] + [
        CPrimitive("GetFirst_"+name+str(i), tnesting, _getfirst(tp, i), (defaultdict(int, {tp:i} ), i+1 if i>=0 else abs(i))) for name, tp in types.items() for i in list(range(-max_index,0))+ list(range(1,max_index+1))
        ] + [ 
        CPrimitive("GetAll_"+name, tnesting, _getall(reg),(defaultdict(int, {reg:1} ),1) ) for name, reg in types.items()
        ] + [
        #regexes
        CPrimitive("type_to_regex", arrow(ttype, tregex), lambda x: x), #TODO also make disappear
        CPrimitive("delimiter_to_regex", arrow(tdelimiter, tregex), lambda x: re.escape(x)) #TODO also make disappear
        ] + [
        #types
        CPrimitive("Number", ttype, r'\d+', r'\d+'), #TODO
        CPrimitive("Word", ttype, r'\w+', r'\w+'), #TODO
        CPrimitive("Alphanum", ttype, r'\w', r'\w'), #TODO
        CPrimitive("PropCase", ttype, r'[A-Z][a-z]+', r'[A-Z][a-z]+'), #TODO
        CPrimitive("AllCaps", ttype, r'[A-Z]', r'[A-Z]'), #TODO
        CPrimitive("Lower", ttype, r'[a-z]', r'[a-z]'), #TODO
        CPrimitive("Digit", ttype, r'\d', r'\d'), #TODO
        CPrimitive("Char", ttype, r'.', r'.') #TODO
        ] + [
        #Cases
        # CPrimitive("ProperCase", tcase, .title()), #TODO
        # CPrimitive("AllCapsCase", tcase, .upper()), #TODO
        # CPrimitive("LowerCase", tcase, .lower()) #TODO
        ] + [
        #positions
        CPrimitive("position"+str(i), tposition, i, (defaultdict(int), i+1 if i>=0 else abs(i)) ) for i in range(-max_len,max_len+1) #deal with indicies 
        ] + [
        #indices
        CPrimitive("index"+str(i), tindex, i, i) for i in range(-max_index,max_index+1) #deal with indicies
        ] + [
        #characters
        CPrimitive(i, tcharacter, i, (defaultdict(int, {i:1}),1) ) for i in printable[:-5] if i not in disallowed
            ] + [
        CPrimitive(name, tcharacter, char, (defaultdict(int, {char:1}), 1)) for char, name in disallowed.items() # NB: disallowed is reversed
        ] + [
        #delimiters
        CPrimitive("delim_"+name, tdelimiter, char, char) for name, char in delim_dict.items()
        ] + [
        #boundaries
        CPrimitive("End", tboundary, "End"),
        CPrimitive("Start", tboundary, "Start")
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
