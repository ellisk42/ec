from program import Primitive, Program
from listPrimitives import bootstrapTarget
from type import arrow, baseType, t0, tint

turtle = baseType("turtle")
tstate = baseType("tstate")
# tint = baseType("tint")

def _logo_var_next(x):
    return "(logo_var_NXT " + x + ")"
def _logo_var_prev(x):
    return "(logo_var_PRV " + x + ")"
def _logo_var_double(x):
    return "(logo_var_DBL " + x + ")"
def _logo_var_half(x):
    return "(logo_var_HLF " + x + ")"
def _logo_var_add(x):
    return lambda v2: "(logo_var_ADD " + x + " " + v2 + ")"
def _logo_var_sub(x):
    return lambda v2: "(logo_var_SUB " + x + " " + v2 + ")"
def _logo_var_div(x):
    return lambda v2: "(logo_var_DIV " + x + " " + v2 + ")"
def _logo_var_mul(x):
    return lambda v2: "(logo_var_MUL " + x + " " + v2 + ")"

def _logo_fw(x):
    return "(logo_FW " + x + ")"
def _logo_rt(x):
    return "(logo_RT " + x + ")"
def _logo_set(x):
    return "(logo_SET " + x + ")"
def _logo_seq(v):
    return lambda v2: "(logo_SEQ " + v + " " + v2 + ")"
def _logo_get(v):
    return lambda v2: "(logo_GET " + v + " " + v2 + ")"

primitives = [
    Primitive("logo_var_UNIT", tint, "logo_var_UNIT"),
    Primitive("logo_var_TWO", tint, "logo_var_TWO"),
    Primitive("logo_var_THREE", tint, "logo_var_THREE"),
    Primitive("logo_var_PI",   tint, "logo_var_PI"),
    Primitive("logo_var_NXT",  arrow(tint,tint), _logo_var_next),
    Primitive("logo_var_PRV",  arrow(tint,tint), _logo_var_prev),
    Primitive("logo_var_DBL",  arrow(tint,tint), _logo_var_double),
    Primitive("logo_var_HLF",  arrow(tint,tint), _logo_var_half),
    Primitive("logo_var_ADD",  arrow(tint,tint,tint), _logo_var_add),
    Primitive("logo_var_SUB",  arrow(tint,tint,tint), _logo_var_sub),
    Primitive("logo_var_DIV",  arrow(tint,tint,tint), _logo_var_div),
    Primitive("logo_var_MUL",  arrow(tint,tint,tint), _logo_var_mul),

    Primitive("logo_NOP", turtle, "logo_NOP"),
    Primitive("logo_PU",  turtle, "logo_PU"),
    Primitive("logo_PD",  turtle, "logo_PD"),
    Primitive("logo_FW",  arrow(tint,turtle), _logo_fw),
    Primitive("logo_RT",  arrow(tint,turtle), _logo_rt),
    Primitive("logo_SET",  arrow(tstate,turtle), _logo_set),
    Primitive("logo_SEQ",  arrow(turtle,turtle,turtle), _logo_seq),
    Primitive("logo_GET",  arrow(tstate,turtle,turtle), _logo_get)
] + bootstrapTarget()

if __name__ == "__main__":
    # x = Program.parse(
        # "(#(concat (integrate nothing nothing nothing nothing)) (turn nothing))")
    # x = Program.parse("(integrate nothing nothing nothing nothing)")
    print((x.evaluate([])))
