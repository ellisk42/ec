from program import Primitive
from listPrimitives import _map, _unfold, _range, _index, _fold, _if, _addition, _subtraction, _cons, _car, _cdr, _isEmpty, bootstrapTarget
from type import arrow, baseType, t0, t1, tint, tlist, tbool

turtle = baseType("turtle")
tstate = baseType("tstate")
tangle = baseType("tangle")
tlength = baseType("tlength")
tscalar = baseType("tscalar")

def _logo_I2S(x):
    return str(float(x))
def _logo_S2A(x):
    return str(x)
def _logo_S2L(x):
    return str(x)

def _logo_var_divs(v1):
    return lambda v2: str(v1/v2)
def _logo_var_muls(v1):
    return lambda v2: str(v1*v2)

def _logo_var_diva(v1):
    return lambda v2: str(v1/v2)
def _logo_var_mula(v1):
    return lambda v2: str(v1*v2)
def _logo_var_divl(v1):
    return lambda v2: str(v1/v2)
def _logo_var_mull(v1):
    return lambda v2: str(v1*v2)

def _logo_var_adda(v1):
    return lambda v2: str(v1+v2)
def _logo_var_suba(v1):
    return lambda v2: str(v1-v2)
def _logo_var_addl(v1):
    return lambda v2: str(v1+v2)
def _logo_var_subl(v1):
    return lambda v2: str(v1-v2)

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

# def _logo_cheat(x):
    # return "(logo_CHEAT " + x + ")"
# def _logo_cheat2(x):
    # return "(logo_CHEAT2 " + x + ")"
# def _logo_cheat3(x):
    # return "(logo_CHEAT3 " + x + ")"
# def _logo_cheat4(x):
    # return "(logo_CHEAT4 " + x + ")"

primitives = [
    Primitive("logo_I2S", arrow(tint,tscalar), _logo_I2S),
    Primitive("logo_S2A", arrow(tscalar,tangle), _logo_S2A),
    Primitive("logo_S2L", arrow(tscalar,tlength), _logo_S2L),

    Primitive("logo_DIVS",  arrow(tscalar,tscalar,tscalar), _logo_var_divs),
    Primitive("logo_MULS",  arrow(tscalar,tscalar,tscalar), _logo_var_muls),

    Primitive("logo_DIVA",  arrow(tangle,tscalar,tangle), _logo_var_diva),
    Primitive("logo_MULA",  arrow(tangle,tscalar,tangle), _logo_var_mula),
    Primitive("logo_DIVL",  arrow(tlength,tscalar,tlength), _logo_var_divl),
    Primitive("logo_MULL",  arrow(tlength,tscalar,tlength), _logo_var_mull),

    Primitive("logo_ADDA",  arrow(tangle,tangle,tangle), _logo_var_adda),
    Primitive("logo_SUBA",  arrow(tangle,tangle,tangle), _logo_var_suba),
    Primitive("logo_ADDL",  arrow(tlength,tlength,tlength), _logo_var_addl),
    Primitive("logo_SUBL",  arrow(tlength,tlength,tlength), _logo_var_subl),

    Primitive("logo_NOP", turtle, "logo_NOP"),
    # Primitive("logo_PU",  turtle, "logo_PU"),
    # Primitive("logo_PD",  turtle, "logo_PD"),
    Primitive("logo_FW",  arrow(tlength,turtle), _logo_fw),
    Primitive("logo_RT",  arrow(tangle,turtle), _logo_rt),
    Primitive("logo_SET",  arrow(tstate,turtle), _logo_set),
    Primitive("logo_SEQ",  arrow(turtle,turtle,turtle), _logo_seq),
    Primitive("logo_GET",  arrow(arrow(tstate,turtle),turtle), _logo_get)

    # Primitive("logo_CHEAT",  arrow(ttvar,turtle), _logo_cheat),
    # Primitive("logo_CHEAT2",  arrow(ttvar,turtle), _logo_cheat),
    # Primitive("logo_CHEAT3",  arrow(ttvar,turtle), _logo_cheat),
    # Primitive("logo_CHEAT4",  arrow(ttvar,turtle), _logo_cheat),
] + [
    Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
    Primitive("range", arrow(tint, tlist(tint)), _range),
    Primitive("fold", arrow(tlist(t0), t1, arrow(t0, t1, t1), t1), _fold),
    Primitive("+", arrow(tint, tint, tint), _addition),
    Primitive("-", arrow(tint, tint, tint), _subtraction),
    Primitive("20", tint, 20),
] + [Primitive(str(j), tint, j) for j in range(7)]

if __name__ == "__main__":
    # x = Program.parse(
        # "(#(concat (integrate nothing nothing nothing nothing)) (turn nothing))")
    # x = Program.parse("(integrate nothing nothing nothing nothing)")
    print((x.evaluate([])))
