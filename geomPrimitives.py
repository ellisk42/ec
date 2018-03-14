from program import Primitive
from type import arrow, baseType, tmaybe, t0

tprogram = baseType("program")
tstring  = baseType("string")
tcanvas  = baseType("canvas")
tvar     = baseType("var")
tbool    = baseType("bool")

primitives = [
    # VAR
    Primitive("var_unit", tvar, None),
    Primitive("var_double", arrow(tvar,tvar), None),
    Primitive("var_half", arrow(tvar,tvar), None),
    Primitive("var_next", arrow(tvar, tvar), None),
    Primitive("var_prev", arrow(tvar, tvar), None),
    Primitive("var_opposite", arrow(tvar, tvar), None),
    # Primitive("var_name", arrow(tstring, tvar), None),

    # PROGRAMS
    Primitive("embed", arrow(tprogram,tprogram), None),
    Primitive("integrate",
              arrow(tmaybe(tvar),
                    tmaybe(tbool),
                    tmaybe(tvar),
                    tmaybe(tvar),
                    tmaybe(tvar),
                    tmaybe(tvar),
                    tprogram), None),
    Primitive("turn", arrow(tmaybe(tvar),tprogram), None),
    Primitive("repeat", arrow(tmaybe(tvar),tprogram,tprogram), None),
    Primitive("concat", arrow(tprogram,tprogram,tprogram), None),

    # RUN
    Primitive("run", arrow(tprogram,tcanvas), None),

    # tbool
    Primitive("true",  tbool, None),
    Primitive("false", tbool, None),

    # maybe
    Primitive("just", arrow(t0,tmaybe(t0)), None),
    Primitive("nothing", tmaybe(t0), None)
]
