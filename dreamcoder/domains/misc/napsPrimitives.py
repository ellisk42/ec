#napsPrimitives.py
from dreamcoder.program import Primitive, prettyProgram
from dreamcoder.grammar import Grammar
from dreamcoder.type import tlist, tint, arrow, baseType #, t0, t1, t2

#from functools import reduce


#types
PROGRAM = baseType("PROGRAM")

RECORD = baseType("RECORD")
FUNC = baseType("FUNC")

VAR = baseType("VAR")
STMT = baseType("STMT")
EXPR = baseType("EXPR")
ASSIGN = baseType("ASSIGN")
LHS = baseType("LHS")
IF = baseType("IF")
FOREACH = baseType("FOREACH")
WHILE = baseType("WHILE")
BREAK = baseType("BREAK")
CONTINUE = baseType("CONTINUE")
RETURN = baseType("RETURN")
NOOP = baseType("NOOP")
FIELD = baseType("FIELD")
CONSTANT = baseType("CONSTANT")
INVOKE = baseType("INVOKE")
TERNARY = baseType("TERNARY")
CAST = baseType("CAST")
TYPE = baseType("TYPE")

#other types
function_name = baseType("function_name")
field_name = baseType("field_name")
name = baseType("name")  # for records and functions
value = baseType("value")

# definitions:

def _program(records): return lambda funcs: {'types': records, 'funcs': funcs}
# record
def _func(string): return lambda tp: lambda name: lambda vars1: lambda vars2: lambda stmts: [string, tp, name, vars1, vars2, stmts]
def _var(tp): return lambda name: ['var', tp, name]
# stmt
# expr
def _assign(tp): return lambda lhs: lambda expr: ['assign', tp, lhs, expr]
# lhs 
def _if(tp): return lambda expr: lambda stmts1: lambda stmts2: ['if', tp, expr, stmts1, stmts2]  # TODO
def _foreach(tp): return lambda var: lambda expr: lambda stmts: ['foreach', tp, expr, stmts]  # TODO
def _while(tp): return lambda expr: lambda stmts1: lambda stmts1: ['while', tp, expr, stmts1, stmts2]  # or: ['while', tp, expr, [stmts1], [stmts2]] #TODO
# break
# continue
def _return(tp): return lambda expr: ['return', tp, expr]
# noop
def _field(tp): return lambda expr: lambda field_name: ['field', tp, expr, field_name]
def _constant(tp): return lambda value: ['val', tp, value]  #TODO deal with value
def _invoke(tp): return lambda function_name: lambda exprs: ['invoke', tp, function_name, exprs]  # TODO, deal with fn_name and lists
def _ternary(tp): return lambda expr1: lambda expr2: lambda expr3: ['?:', tp, expr1, expr2, expr3]
def _cast(tp): return lambda expr: ['cast', tp, expr]

# types:

# TODO: deal with lists - x 
# TODO: deal with names
# TODO: deal with values - x 

# TODO: deal with the program/record __main__ and __globals__ stuff



def napsPrimitives():
    return [
        Primitive("program", arrow(tlist(RECORD), tlist(FUNC), PROGRAM), _program), # TODO
        # RECORD
        Primitive("func", arrow(TYPE, name, tlist(VAR), tlist(VAR), tlist(VAR), tlist(STMT)), _func('func')), # TODO
        Primitive("ctor", arrow(TYPE, name, tlist(VAR), tlist(VAR), tlist(VAR), tlist(STMT)), _func('ctor')),
        Primitive("var", arrow(TYPE, name, VAR), _var)
        ] + [ 
        # STMT ::= EXPR | IF | FOREACH | WHILE | BREAK | CONTINUE | RETURN | NOOP
        Primitive("stmt_expr", arrow(EXPR, STMT), lambda x: x),
        Primitive("stmt_if", arrow(IF, STMT), lambda x: x),
        Primitive("stmt_foreach", arrow(FOREACH, STMT), lambda x: x),
        Primitive("stmt_while", arrow(WHILE, STMT), lambda x: x),
        Primitive("stmt_break", arrow(BREAK, STMT), lambda x: x),
        Primitive("stmt_continue", arrow(CONTINUE, STMT), lambda x: x),
        Primitive("stmt_return", arrow(RETURN, STMT), lambda x: x),
        Primitive("stmt_noop", arrow(NOOP, STMT), lambda x: x)
        ] + [
        # EXPR ::= ASSIGN | VAR | FIELD | CONSTANT | INVOKE | TERNARY | CAST
        Primitive("expr_assign", arrow(ASSIGN, EXPR), lambda x: x),
        Primitive("expr_var", arrow(VAR, EXPR), lambda x: x),
        Primitive("expr_field", arrow(FIELD, EXPR), lambda x: x),
        Primitive("expr_constant", arrow(CONSTANT, EXPR), lambda x: x),
        Primitive("expr_invoke", arrow(INVOKE, EXPR), lambda x: x),
        Primitive("expr_ternary", arrow(TERNARY, EXPR), lambda x: x),
        Primitive("expr_cast", arrow(CAST, EXPR), lambda x: x)
        ] + [
        Primitive("assign", arrow(TYPE, LHS, EXPR, ASSIGN), _assign)
        ] + [
        # LHS ::= VAR | FIELD | INVOKE
        Primitive("lhs_var", arrow(VAR, LHS), lambda x: x),
        Primitive("lhs_field", arrow(FIELD, LHS), lambda x: x),
        Primitive("lhs_invoke", arrow(INVOKE, LHS), lambda x: x)
        ] + [
        Primitive("if", arrow(TYPE, EXPR, tlist(STMT), tlist(STMT), IF), _if),
        Primitive("foreach", arrow(TYPE, VAR, EXPR, tlist(STMT), FOREACH), _foreach),
        Primitive("while", arrow(TYPE, EXPR, tlist(STMT), tlist(STMT), WHILE), _while),
        Primitive("break", arrow(TYPE, BREAK), lambda tp: ['break', tp]),
        Primitive("continue", arrow(TYPE, CONTINUE), lambda tp: ['continue', tp]),
        Primitive("return", arrow(TYPE, EXPR, RETURN), _return),
        Primitive("noop", NOOP, ['noop']),
        Primitive("field", arrow(TYPE, EXPR, field_name, FIELD), _field),  # TODO
        Primitive("constant", arrow(TYPE, value, CONSTANT), _constant),
        Primitive("invoke", arrow(TYPE, function_name, tlist(EXPR), INVOKE), _invoke),  # TODO
        Primitive("ternary", arrow(TYPE, EXPR, EXPR, EXPR, TERNARY), _ternary),
        Primitive("cast", arrow(TYPE, EXPR, CAST), _cast)
        ] + [
        # below are TYPE:
        Primitive("bool", TYPE, 'bool'),
        Primitive("char", TYPE, 'char'),
        Primitive("char*", TYPE, 'char*'),
        Primitive("int", TYPE, 'int'),
        Primitive("real", TYPE, 'real'),
        Primitive("array", arrow(TYPE, TYPE), lambda tp: tp + '*'),
        Primitive("set", arrow(TYPE, TYPE), lambda tp: tp + '%'),
        Primitive("map", arrow(TYPE, TYPE, TYPE), lambda tp1: lambda tp2: '<'+tp1+'|'+tp2+'>'),
        Primitive("record_name", TYPE, 'record_name#')  # TODO
        ] + [
        #stuff about lists:
        # STMTs, EXPRs, VARs, maybe Funcs and records
        Primitive('list_init_stmt', arrow(STMT, tlist(STMT)), lambda stmt: [stmt]),
        Primitive('list_add_stmt', arrow(STMT, tlist(STMT), tlist(STMT)), lambda stmt: lambda stmts: stmts + [stmt]),
        Primitive('list_init_expr', arrow(EXPR, tlist(EXPR)), lambda expr: [expr]),
        Primitive('list_add_expr', arrow(EXPR, tlist(EXPR), tlist(EXPR)), lambda expr: lambda exprs: exprs + [expr]),
        Primitive('list_init_var', arrow(VAR, tlist(VAR)), lambda var: [var]),
        Primitive('list_add_var', arrow(VAR, tlist(VAR), tlist(VAR)), lambda var: lambda _vars: _vars + [var])
        ] + [
        # value
        Primitive('0', value, 0),
        Primitive("1", value, "1"),
        Primitive("-1", value, "-1")
        # ...  
        ] + [
        # function_name:
        Primitive('+', function_name, '+'),
        Primitive('&&', function_name, "&&"),
        Primitive("!", function_name, "!"),
        Primitive("!=", function_name, "!="),
        Primitive("string_find", function_name,"string_find")
        # ... 
        ] + [
        # field_name:
        Primitive('', field_name, '')
        # ...
        ] + [
        # 
        Primitive(f'var{str(i)}', name, f'var{str(i)}') for i in range(12)
    ]


#for first pass, can just hard code vars and maps n stuff

def ec_prog_to_uast(prog):  # TODO
    # ideally, just evaluate and then parse
    uast = prog.evaluate([])
    return uast

def deepcoderProductions():
    return [(0.0, prim) for prim in deepcoderPrimitives()]

# def flatten_program(p):
#     string = p.show(False)
#     num_inputs = string.count('lambda')
#     string = string.replace('lambda', '')
#     string = string.replace('(', '')
#     string = string.replace(')', '')
#     #remove '_fn' (optional)
#     for i in range(num_inputs):
#         string = string.replace('$' + str(num_inputs-i-1),'input_' + str(i))
#     string = string.split(' ')
#     string = list(filter(lambda x: x is not '', string))
#     return string

if __name__ == "__main__":
    #g = Grammar.uniform(deepcoderPrimitives())
    g = Grammar.fromProductions(deepcoderProductions(), logVariable=.9)
    request = arrow(tlist(tint), tint, tint)
    p = g.sample(request)
    print("request:", request)
    print("program:")
    print(prettyProgram(p))
    print("flattened_program:")
    flat = flatten_program(p)
    print(flat)

  
