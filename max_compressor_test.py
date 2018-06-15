#max testing the compressor, to get the stuff I want
from program import Primitive, Program
from grammar import Grammar
from type import tlist, tint, tbool, arrow, t0, t1, t2, tpregex
from string import printable
#from pregex import pregex

import math

# evaluation to regular regex form. then I can unflatten using Luke's stuff.


def _kleene(x): return pregex.KleeneStar(x, p=0.25)


def _plus(x): return pregex.Plus(x, p=0.25)


def _maybe(x): return pregex.Maybe(x)


# maybe should be reversed#"(" + x + "|" + y + ")"
def _alt(x): return lambda y: pregex.Alt([x, y])


def _concat(x): return lambda y: pregex.Concat([x, y])  # "(" + x + y + ")"
#def _wrapper(x): return lambda y: y

#specials = [".","*","+","?","|"]


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

disallowed_list = [char for char, _ in disallowed]

def altPrimitives():
    return [
        Primitive("empty_string", tpregex, None)
    ] + [
        Primitive("string_" + i, tpregex, None) for i in printable[:-4] if i not in disallowed_list
    ] + [
        Primitive("string_" + name, tpregex, None) for char, name in disallowed
    ] + [
        Primitive("r_dot", tpregex, None),
        Primitive("r_d", tpregex, None),
        Primitive("r_s", tpregex, None),
        Primitive("r_w", tpregex, None),
        Primitive("r_l", tpregex, None),
        Primitive("r_u", tpregex, None),
        Primitive("r_kleene", arrow(tpregex, tpregex), None),
        #Primitive("r_plus", arrow(tpregex, tpregex), _plus),
        Primitive("r_maybe", arrow(tpregex, tpregex), None),
        Primitive("r_alt", arrow(tpregex, tpregex, tpregex), None),
        Primitive("r_concat", arrow(tpregex, tpregex, tpregex), None),
    ]

from grammar import *


prim_list = altPrimitives()
n_base_prim = len(prim_list) - 5.
specials = ["r_kleene", "r_plus", "r_maybe", "r_alt", "r_concat"]

productions = [
    (0.25 / n_base_prim,
     prim) if prim.name not in specials else (
        0.15,
        prim) for prim in prim_list]


baseGrammar = Grammar.fromProductions(productions)

#for testing stuff
from program import *
from frontier import Frontier
from fragmentGrammar import *

frontiers = []

program1 = Program.parse("(r_concat r_dot (r_kleene r_dot))")
frontier = Frontier.dummy(program1, logLikelihood=0., logPrior=0.)
frontiers.append(frontier)

program2 = Program.parse("(r_concat r_d (r_kleene r_d))" )
frontier = Frontier.dummy(program2, logLikelihood=0., logPrior=0.)
frontiers.append(frontier)

program3 = Program.parse("(r_concat r_u (r_kleene r_u))" )
frontier = Frontier.dummy(program3, logLikelihood=0., logPrior=0.)
frontiers.append(frontier)

program4 = Program.parse("(r_concat r_w (r_kleene r_w))" )
frontier = Frontier.dummy(program4, logLikelihood=0., logPrior=0.)
frontiers.append(frontier)


#grammar, frontiers = induceGrammar(baseGrammar, frontiers,
#                           topK=5, topk_use_map=False,
#                           pseudoCounts=1.0, a=3,
#                           aic=0.0, structurePenalty=.1,
#                           backend='rust', CPUs=1, iteration=1)

grammar, frontiers = induceGrammar(baseGrammar, frontiers,
                                   topK=5,
                                   pseudoCounts=1.0, a=3,
                                   aic=0.0, structurePenalty=1,
                                   topk_use_only_likelihood=True,
                                   backend='rust', CPUs=1, iteration=1)
