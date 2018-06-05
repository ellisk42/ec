from program import Primitive, Program
from grammar import Grammar
from type import tlist, tint, tbool, arrow, t0, t1, t2, tpregex
from string import printable
from pregex import pregex

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


def basePrimitives():
    return [Primitive("string_" + i, tpregex, pregex.String(i)) for i in printable[:-4] if i not in disallowed_list
            ] + [
        Primitive("string_" + name, tpregex, pregex.String(char)) for char, name in disallowed
    ] + [
        Primitive("r_dot", tpregex, pregex.dot),
        Primitive("r_d", tpregex, pregex.d),
        Primitive("r_s", tpregex, pregex.s),
        Primitive("r_w", tpregex, pregex.w),
        Primitive("r_l", tpregex, pregex.l),
        Primitive("r_u", tpregex, pregex.u),
        Primitive("r_kleene", arrow(tpregex, tpregex), _kleene),
        Primitive("r_plus", arrow(tpregex, tpregex), _plus),
        Primitive("r_maybe", arrow(tpregex, tpregex), _maybe),
        Primitive("r_alt", arrow(tpregex, tpregex, tpregex), _alt),
        Primitive("r_concat", arrow(tpregex, tpregex, tpregex), _concat),
    ]



def altPrimitives():
    return [
        Primitive("empty_string", tpregex, pregex.String(""))
    ] + [
        Primitive("string_" + i, tpregex, pregex.String(i)) for i in printable[:-4] if i not in disallowed_list
    ] + [
        Primitive("string_" + name, tpregex, pregex.String(char)) for char, name in disallowed
    ] + [
        Primitive("r_dot", tpregex, pregex.dot),
        Primitive("r_d", tpregex, pregex.d),
        Primitive("r_s", tpregex, pregex.s),
        Primitive("r_w", tpregex, pregex.w),
        Primitive("r_l", tpregex, pregex.l),
        Primitive("r_u", tpregex, pregex.u),
        Primitive("r_kleene", arrow(tpregex, tpregex), _kleene),
        #Primitive("r_plus", arrow(tpregex, tpregex), _plus),
        Primitive("r_maybe", arrow(tpregex, tpregex), _maybe),
        Primitive("r_alt", arrow(tpregex, tpregex, tpregex), _alt),
        Primitive("r_concat", arrow(tpregex, tpregex, tpregex), _concat),
    ]