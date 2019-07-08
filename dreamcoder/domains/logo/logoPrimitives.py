from dreamcoder.program import Primitive, Program
from dreamcoder.type import arrow, baseType, tint

turtle = baseType("turtle")
tstate = baseType("tstate")
tangle = baseType("tangle")
tlength = baseType("tlength")

primitives = [
    Primitive("logo_UA", tangle, ""),
    Primitive("logo_UL", tlength, ""),

    Primitive("logo_ZA", tangle, ""),
    Primitive("logo_ZL", tlength, ""),

    Primitive("logo_DIVA",  arrow(tangle,tint,tangle), ""),
    Primitive("logo_MULA",  arrow(tangle,tint,tangle), ""),
    Primitive("logo_DIVL",  arrow(tlength,tint,tlength), ""),
    Primitive("logo_MULL",  arrow(tlength,tint,tlength), ""),

    Primitive("logo_ADDA",  arrow(tangle,tangle,tangle), ""),
    Primitive("logo_SUBA",  arrow(tangle,tangle,tangle), ""),
    # Primitive("logo_ADDL",  arrow(tlength,tlength,tlength), ""),
    # Primitive("logo_SUBL",  arrow(tlength,tlength,tlength), ""),

    # Primitive("logo_PU",  arrow(turtle,turtle), ""),
    # Primitive("logo_PD",  arrow(turtle,turtle), ""),
    Primitive("logo_PT", arrow(arrow(turtle,turtle),arrow(turtle,turtle)), None),
    Primitive("logo_FWRT",  arrow(tlength,tangle,turtle,turtle), ""),
    Primitive("logo_GETSET",  arrow(arrow(turtle,turtle),turtle,turtle), "")
] + [
    Primitive("logo_IFTY", tint, ""),
    Primitive("logo_epsA", tangle, ""),
    Primitive("logo_epsL", tlength, ""),
    Primitive("logo_forLoop", arrow(tint, arrow(tint, turtle, turtle), turtle, turtle), "ERROR: python has no way of expressing this hence you shouldn't eval on this"),
] + [Primitive(str(j), tint, j) for j in range(10)]

if __name__ == "__main__":
    expr_s = "(lambda (logo_forLoop 3 (lambda (lambda (logo_GET (lambda (logo_FWRT (logo_S2L (logo_I2S 1)) (logo_S2A (logo_I2S 0)) (logo_SET $0 (logo_FWRT (logo_S2L eps) (logo_DIVA (logo_S2A (logo_I2S 2)) (logo_I2S 3)) ($1)))))))) ($0)))"
    x = Program.parse(expr_s)
    print(x)
