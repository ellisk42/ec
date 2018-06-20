from program import Primitive
from listPrimitives import _map, _unfold, _range, _index, _fold, _if, _addition, _subtraction, _cons, _car, _cdr, _isEmpty, bootstrapTarget
from type import arrow, baseType, t0, t1, tint, tlist, tbool

turtle = baseType("turtle")
tstate = baseType("tstate")
ttvar = baseType("ttvar")

def _logo_var_next(x):
    return "(logo_var_NXT " + x + ")"
def _logo_var_prev(x):
    return "(logo_var_PRV " + x + ")"
def _logo_var_double(x):
    return "(logo_var_DBL " + x + ")"
def _logo_var_half(x):
    return "(logo_var_HLF " + x + ")"

def _logo_F2I(x):
    return "(logo_F2I " + str(x) + ")"
def _logo_I2F(x):
    return "(logo_I2F " + str(x) + ")"

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
def _logo_cheat(x):
    return "(logo_CHEAT " + x + ")"
def _logo_cheat2(x):
    return "(logo_CHEAT2 " + x + ")"
def _logo_cheat3(x):
    return "(logo_CHEAT3 " + x + ")"
def _logo_rt(x):
    return "(logo_RT " + x + ")"
def _logo_set(x):
    return "(logo_SET " + x + ")"
def _logo_seq(v):
    return lambda v2: "(logo_SEQ " + v + " " + v2 + ")"
def _logo_get(v):
    return lambda v2: "(logo_GET " + v + " " + v2 + ")"

primitives = [
    Primitive("logo_var_UNIT", ttvar, "logo_var_UNIT"),
    Primitive("logo_var_IFTY", ttvar, "logo_var_IFTY"),
    Primitive("logo_var_TWO", ttvar, "logo_var_TWO"),
    Primitive("logo_var_THREE", ttvar, "logo_var_THREE"),
    # Primitive("logo_var_PI",   ttvar, "logo_var_PI"),
    Primitive("logo_var_NXT",  arrow(ttvar,ttvar), _logo_var_next),
    Primitive("logo_var_PRV",  arrow(ttvar,ttvar), _logo_var_prev),
    Primitive("logo_var_DBL",  arrow(ttvar,ttvar), _logo_var_double),
    Primitive("logo_var_HLF",  arrow(ttvar,ttvar), _logo_var_half),
    Primitive("logo_var_ADD",  arrow(ttvar,ttvar,ttvar), _logo_var_add),
    Primitive("logo_var_SUB",  arrow(ttvar,ttvar,ttvar), _logo_var_sub),
    Primitive("logo_var_DIV",  arrow(ttvar,ttvar,ttvar), _logo_var_div),
    Primitive("logo_var_MUL",  arrow(ttvar,ttvar,ttvar), _logo_var_mul),

    Primitive("logo_NOP", turtle, "logo_NOP"),
    Primitive("logo_PU",  turtle, "logo_PU"),
    Primitive("logo_PD",  turtle, "logo_PD"),
    Primitive("logo_FW",  arrow(ttvar,turtle), _logo_fw),
    Primitive("logo_CHEAT",  arrow(ttvar,turtle), _logo_cheat),
    Primitive("logo_CHEAT2",  arrow(ttvar,turtle), _logo_cheat),
    Primitive("logo_CHEAT3",  arrow(ttvar,turtle), _logo_cheat),
    Primitive("logo_RT",  arrow(ttvar,turtle), _logo_rt),
    Primitive("logo_SET",  arrow(tstate,turtle), _logo_set),
    Primitive("logo_SEQ",  arrow(turtle,turtle,turtle), _logo_seq),
    Primitive("logo_GET",  arrow(tstate,turtle,turtle), _logo_get),

    Primitive("logo_I2F", arrow(tint,ttvar), _logo_I2F),
    Primitive("logo_F2I", arrow(ttvar,tint), _logo_I2F)

] + bootstrapTarget() + [Primitive(str(j), tint, j) for j in range(2,7)]
# ] + [
    # Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
    # Primitive("zip", arrow(tlist(t0), tlist(t1), arrow(t0, t1, t2), tlist(t2)), _zip),
    # Primitive("unfold", arrow(t0, arrow(t0,tbool), arrow(t0,t1), arrow(t0,t0), tlist(t1)), _unfold),
    # Primitive("range", arrow(tint, tlist(tint)), _range),
    # Primitive("index", arrow(tint, tlist(t0), t0), _index),
    # Primitive("fold", arrow(tlist(t0), t1, arrow(t0, t1, t1), t1), _fold),
    # Primitive("length", arrow(tlist(t0), tint), len),
    # Primitive("if", arrow(tbool, t0, t0, t0), _if),
    # Primitive("+", arrow(tint, tint, tint), _addition),
    # Primitive("-", arrow(tint, tint, tint), _subtraction),
    # Primitive("empty", tlist(t0), []),
    # Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), _cons),
    # Primitive("car", arrow(tlist(t0), t0), _car),
    # Primitive("cdr", arrow(tlist(t0), tlist(t0)), _cdr),
    # Primitive("empty?", arrow(tlist(t0), tbool), _isEmpty),
# ] + [Primitive(str(j), tint, j) for j in range(3)]

if __name__ == "__main__":
    # x = Program.parse(
        # "(#(concat (integrate nothing nothing nothing nothing)) (turn nothing))")
    # x = Program.parse("(integrate nothing nothing nothing nothing)")
    print((x.evaluate([])))
