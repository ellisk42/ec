from program import Primitive, Program
from type import arrow, baseType, tmaybe, t0

tprogram = baseType("program")
tstring  = baseType("string")
tcanvas  = baseType("canvas")
tvar     = baseType("var")
tbool    = baseType("bool")


def _var_double(x):
    return "(var_double " + x + ")"
def _var_half(x):
    return "(var_half " + x + ")"
def _var_next(x):
    return "(var_next " + x + ")"
def _var_prev(x):
    return "(var_prev " + x + ")"
def _var_opposite(x):
    return "(var_opposite " + x + ")"

def _embed(p):
    return "(embed " + p + ")"
def _turn(p):
    return "(turn " + p + ")"
def _run(p):
    return "(run " + p + ")"
def _define(v):
    return "(define " + v + ")"
def _just(v):
    return "(just " + v + ")"
def _integrate(v1):
    return lambda v2: \
           lambda v3: \
           lambda v4: \
           "(integrate" + \
           " " + v1 + \
           " " + v2 + \
           " " + v3 + \
           " " + v4 + \
           ")"
def _repeat(v):
    return lambda p: "(repeat " + v + " " + p + ")"
def _concat(p1):
    return lambda p2: "(concat " + p1 + " " + p2 + ")"


primitives = [
    # VAR
    Primitive("var_unit", tvar, "var_unit"),
    Primitive("var_double", arrow(tvar, tvar), _var_double),
    Primitive("var_half", arrow(tvar, tvar), _var_half),
    Primitive("var_next", arrow(tvar, tvar), _var_next),
    Primitive("var_prev", arrow(tvar, tvar), _var_prev),
    Primitive("var_opposite", arrow(tvar, tvar), _var_opposite),
    Primitive("var_name", tvar, "var_name"),

    # PROGRAMS
    # Primitive("nop", tprogram, "nop"),
    # Primitive("nop2", tprogram, "nop2"),
    Primitive("embed", arrow(tprogram, tprogram), _embed),
    Primitive("define", arrow(tvar, tprogram), _define),
    Primitive("integrate",
              arrow(tmaybe(tvar),
                    tbool,
                    tmaybe(tvar),
                    tmaybe(tvar),
                    tprogram), _integrate),
    Primitive("turn", arrow(tmaybe(tvar), tprogram), _turn),
    Primitive("repeat", arrow(tmaybe(tvar), tprogram, tprogram), _repeat),
    Primitive("concat", arrow(tprogram, tprogram, tprogram), _concat),

    # RUN
    Primitive("run", arrow(tprogram, tcanvas), _run),

    # tbool
    Primitive("true",  tbool, "true"),
    Primitive("false", tbool, "false"),

    # maybe
    Primitive("just", arrow(t0, tmaybe(t0)), _just),
    Primitive("nothing", tmaybe(t0), "nothing")
]

if __name__ == "__main__":
    x = Program.parse("(#(concat (integrate nothing nothing nothing nothing)) (turn nothing))")
    # x = Program.parse("(integrate nothing nothing nothing nothing)")
    print(x.evaluate([]))
