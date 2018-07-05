from program import Primitive, Program
from listPrimitives import _map, _unfold, _range, _index, _fold, _if, _addition, _subtraction, _cons, _car, _cdr, _isEmpty, bootstrapTarget
from type import arrow, baseType, t0, t1, tint, tlist, tbool

turtle = baseType("turtle")
tstate = baseType("tstate")
tangle = baseType("tangle")
tlength = baseType("tlength")
tscalar = baseType("tscalar")

def _logo_I2S(x):
    return float(x)
def _logo_S2A(x):
    return x
def _logo_S2L(x):
    return x

def _logo_var_divs(v1):
    return lambda v2: v1/v2
def _logo_var_muls(v1):
    return lambda v2: v1*v2

def _logo_var_diva(v1):
    return lambda v2: v1/v2
def _logo_var_mula(v1):
    return lambda v2: v1*v2
def _logo_var_divl(v1):
    return lambda v2: v1/v2
def _logo_var_mull(v1):
    return lambda v2: v1*v2

def _logo_var_adda(v1):
    return lambda v2: v1+v2
def _logo_var_suba(v1):
    return lambda v2: v1-v2
def _logo_var_addl(v1):
    return lambda v2: v1+v2
def _logo_var_subl(v1):
    return lambda v2: v1-v2

# def _logo_fw(x):
    # return "(logo_FW " + str(x) + ")"
# def _logo_rt(x):
    # return "(logo_RT " + str(x) + ")"
def _logo_line(v1):
    return "(line " + v1 + ")"
def _logo_fwrt(v1):
    return lambda v2: lambda v3: "(logo_FWRT " + str(v1) + " " + str(v2) + " " + v3 + ")"
def _logo_set(x):
    return "(logo_SET " + x + ")"
# def _logo_seq(v):
    # return lambda v2: "(logo_SEQ " + str(v) + " " + str(v2) + ")"
def _logo_get(v):
    return lambda v2: "(logo_GET " + str(v) + " " + str(v2) + ")"

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

    # Primitive("logo_DIVS",  arrow(tscalar,tscalar,tscalar), _logo_var_divs),
    # Primitive("logo_MULS",  arrow(tscalar,tscalar,tscalar), _logo_var_muls),

    Primitive("logo_DIVA",  arrow(tangle,tscalar,tangle), _logo_var_diva),
    # Primitive("logo_MULA",  arrow(tangle,tscalar,tangle), _logo_var_mula),
    # Primitive("logo_DIVL",  arrow(tlength,tscalar,tlength), _logo_var_divl),
    # Primitive("logo_MULL",  arrow(tlength,tscalar,tlength), _logo_var_mull),

    # Primitive("logo_ADDA",  arrow(tangle,tangle,tangle), _logo_var_adda),
    # Primitive("logo_SUBA",  arrow(tangle,tangle,tangle), _logo_var_suba),
    # Primitive("logo_ADDL",  arrow(tlength,tlength,tlength), _logo_var_addl),
    # Primitive("logo_SUBL",  arrow(tlength,tlength,tlength), _logo_var_subl),

    # Primitive("logo_NOP", turtle, "logo_NOP"),
    # Primitive("logo_PU",  arrow(turtle,turtle), "logo_PU"),
    # Primitive("logo_PD",  arrow(turtle,turtle), "logo_PD"),
    # Primitive("logo_FW",  arrow(tlength,turtle), _logo_fw),
    # Primitive("logo_RT",  arrow(tangle,turtle), _logo_rt),
    Primitive("logo_FWRT",  arrow(tlength,tangle,turtle,turtle), _logo_fwrt),
    Primitive("logo_SET",  arrow(tstate,turtle,turtle), _logo_set),
    # Primitive("logo_SEQ",  arrow(turtle,turtle,turtle), _logo_seq),
    Primitive("logo_GET",  arrow(arrow(tstate,turtle),turtle), _logo_get)
    # Primitive("logo_GET",  arrow(arrow(tstate,turtle),turtle,turtle), _logo_get)

    # Primitive("logo_CHEAT",  arrow(ttvar,turtle), _logo_cheat),
    # Primitive("logo_CHEAT2",  arrow(ttvar,turtle), _logo_cheat),
    # Primitive("logo_CHEAT3",  arrow(ttvar,turtle), _logo_cheat),
    # Primitive("logo_CHEAT4",  arrow(ttvar,turtle), _logo_cheat),
] + [
    Primitive("ifty", tint, 20),
    Primitive("eps", tscalar, 0.05),
    Primitive("line", arrow(turtle, turtle), _logo_line),
    Primitive("logo_forLoop", arrow(tint, arrow(tint, turtle, turtle), turtle, turtle), "ERROR: python has no way of expressing this hence you shouldn't eval on this"),
] + [Primitive(str(j), tint, j) for j in range(7)]

if __name__ == "__main__":
    x = Program.parse("(lambda (fold #(range 20) $0 (lambda (lambda (line (logo_FWRT (logo_S2L (eps)) (logo_S2A (eps)) $0))))))")
    print(x)


expr_s = "(lambda (lambda (lambda (fold (range $1) $0 (lambda (lambda ($4 $1 $0)))))))"
