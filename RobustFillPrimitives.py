#RobustFillPrimitives

from program import Primitive, Program, prettyProgram
from grammar import Grammar
from type import tlist, tint, tbool, arrow, baseType #, t0, t1, t2

import math
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
delim_dict = {d[c]:c for c in delimiters}

types = {}
types["Number"] = r'\d+'
types["Word"] = r'\w+'
types["Alphanum"] = r'\w'
types["PropCase"] = r'[A-Z][a-z]+'
types["AllCaps"] = r'[A-Z]'
types["Lower"] = r'[a-z]'
types["Digit"] = r'\d'
types["Char"] = r'.' 

regexes = {name: re.escape(val) for name, val in delim_dict.items(), **types}

tstring = baseType("string")
tposition = baseType("tposition")
tindex = baseType("index")
tcharacter = baseType("character")
tboundary = baseType("boundary")
tregex = baseType("tregex")

def _substr(k1): return lambda k2: lambda string: string[k1:k2] #i think this is fine
def _getspan(r1): 
    return 
    lambda i1:
    lambda b1:
    lambda r2:
    lambda i2:
    lambda b2:
    lambda string:
    a = [m.end() for m in re.finditer(r1, string)][i1] if b1 == "End" else [m.start() for m in re.finditer(r1, string)][i1]
    b = [m.end() for m in re.finditer(r2, string)][i2] if b2 == "End" else [m.start() for m in re.finditer(r2, string)][i2]
    string[a:b]
    #TODO format correctly

def _trim(string): assert False return #TODO

def _replace(d1, d2): return lambda string: string.replace(d1,d2)

def _getall(tp): return lambda string: ''.join(re.findall(tp, string))
def _getfirst(tp, i): return lambda string: ''.join(re.findall(tp, string)[:i])
def _gettoken(tp, i): return lambda string: re.findall(tp, string)[i]
def _getupto(reg): return lambda string: string[:[m.end() for m in re.finditer(reg, string)][0]]
def _getfrom(reg): return lambda string: string[[m.end() for m in re.finditer(reg, string)][-1]:]

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
"""


def robustFillPrimitives():
    return [
        #expressions
        Primitive("Constant", arrow(tcharacter,tstring), lambda x: x)
        ] + [
        #substrings
        Primitive("SubStr", arrow(tposition, tposition, tstring, tstring), _substr)
        Primitive("GetSpan", arrow(tregex, tindex, tboundary, tregex, tindex, tboundary, tstring, tstring), _getspan) #TODO
        ] + [
        #nestings
        Primitive("GetToken"+name+str(i), arrow(tstring, tstring), _gettoken(tp,i) ) for name, tp in types.items() for i in range(-5, 5)
        ] + [
        Primitive("ToCase_ProperCase", arrow(tstring, tstring), lambda x: x.title()),
        Primitive("ToCase_AllCapsCase", arrow(tstring, tstring), lambda x: x.upper()),
        Primitive("ToCase_LowerCase", arrow(tstring, tstring), lambda x: x.lower())
        ] + [
        Primitive("Replace_" + name1 + name2, arrow(tstring,tstring), _replace(char1, char2)) for name1, char1 in delim_dict.items() for name2, char2 in delim_dict.items() if char1 is not char2
        ] + [
        Primitive("Trim", arrow(tstring, tstring), _trim), #TODO
        ] + [
        Primitive("GetUpTo"+name, arrow(tstring, tstring), _getupto(reg)) for name, reg in regexes.items()
        ] + [
        Primitive("GetFrom"+name, arrow(tstring, tstring), _getfrom(reg)) for name, reg in regexes.items(), 
        ] + [
        Primitive("GetFirst_" + name + str(i), arrow(tstring, tstring), _getfirst(tp, i) ) for name, tp in types.items() for i in list(range(-5,0))+ list(range(1,6))
        ] + [ 
        Primitive("GetAll_" + name, arrow(tstring, tstring), _getall(reg)) for name, reg in types.items()
        ] + [
        #regexes
        Primitive("type_to_regex", arrow(ttype, tregex), lambda x: x), #TODO also make disappear
        Primitive("delimiter_to_regex", arrow(tdelimiter, tregex), lambda x: re.escape(x)) #TODO also make disappear
        ] + [
        #types
        Primitive("Number", ttype, r'\d+'), #TODO
        Primitive("Word", ttype, r'\w+'), #TODO
        Primitive("Alphanum", ttype, r'\w'), #TODO
        Primitive("PropCase", ttype, r'[A-Z][a-z]+'), #TODO
        Primitive("AllCaps", ttype, r'[A-Z]'), #TODO
        Primitive("Lower", ttype, r'[a-z]'), #TODO
        Primitive("Digit", ttype, r'\d'), #TODO
        Primitive("Char", ttype, r'.') #TODO
        ] + [
        #Cases
        # Primitive("ProperCase", tcase, .title()), #TODO
        # Primitive("AllCapsCase", tcase, .upper()), #TODO
        # Primitive("LowerCase", tcase, .lower()) #TODO
        ] + [
        #positions
        Primitive("position"+i, tposition, i) for i in range(-100,101) #deal with indicies 
        ] + [
        #indices
        Primitive("index"+i, tindex, i) for i in range(-5,6) #deal with indicies
        ] + [
        #characters
        Primitive(i, tcharacter, i) for i in printable[:-5] if i not in disallowed
            ] + [
        Primitive(name, tcharacter, char) for char, name in disallowed.items()
        ] + [
        #delimiters
        Primitive("delim_"+v, tdelimiter, k) for char, name in delim_dict.items()
        ] + [
        #boundaries
        Primitive("End", tboundary, "End"),
        Primitive("Start", tboundary, "Start")
    ]


def RobustFillProductions():
    return [(0.0, prim) for prim in robustFillPrimitives()]


def flatten_program(p):
	# TODO
	return None