# =========== [NEWER VERSION, NOT USING MATPLOTLIB]
import math
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.ndimage import gaussian_filter as gf
from skimage import color
from scipy.ndimage import gaussian_filter as gf
import cairo

from dreamcoder.program import Primitive, Program
from dreamcoder.utilities import Curried
from dreamcoder.grammar import Grammar
from dreamcoder.type import baseType, arrow, tmaybe, t0, t1, t2

from dreamcoder.domains.draw.primitives import *
from dreamcoder.domains.draw.primitives import _makeAffine, _tform, _reflect, _repeat, _connect, _line, _circle

matplotlib.use('TkAgg')

# ======= DEFINE ALL PRIMITIVES
tstroke = baseType("tstroke")
tangle = baseType("tangle")
tscale = baseType("tscale")
tdist = baseType("tdist")
ttrorder = baseType("ttorder")
ttransmat = baseType("ttransmat")
trep = baseType("trep")


def _givemeback(thing):
	return thing

Primitive("None", tmaybe(t0), None)
Primitive("Some", arrow(t0, tmaybe(t0)), _givemeback)


p1 = [
	Primitive("line", tstroke, _line), 
	Primitive("circle", tstroke, _circle),
	Primitive("transmat", arrow(tmaybe(tscale), tmaybe(tangle), tmaybe(tdist), tmaybe(tdist), tmaybe(ttrorder), ttransmat), Curried(_makeAffine)),
	Primitive("transform", arrow(tstroke, ttransmat, tstroke), Curried(_tform)),
	Primitive("reflect", arrow(tstroke, tangle, tstroke), Curried(_reflect)), 
	Primitive("connect", arrow(tstroke, tstroke, tstroke), Curried(_connect)),
	Primitive("repeat", arrow(tstroke, trep, ttransmat, tstroke), Curried(_repeat))
]
# p2 = [Primitive("scale{}".format(i), tscale, j) for i, j in enumerate(np.linspace(1.0, 4.0, 7))]
p2 = [Primitive("scale{}".format(i), tscale, j) for i, j in enumerate(SCALES)] 
# p3 = [Primitive("dist{}".format(i), tdist, j) for i, j in enumerate(np.linspace(-4, 4, 9))]
p3 = [Primitive("dist{}".format(i), tdist, j) for i, j in enumerate(DISTS)]
NANGLE = 8
# p4 = [Primitive("angle{}".format(i), tangle, j*2*math.pi/NANGLE) for i, j in enumerate(range(NANGLE))]
p4 = [Primitive("angle{}".format(i), tangle, j) for i, j in enumerate(THETAS)]
# p5 = [Primitive("angle{}".format(i), tangle, (j+1)*2*math.pi/3) for j,i in enumerate(range(NANGLE, NANGLE+2))]
p5 = []
p6 = [Primitive(j, ttrorder, j) for j in ["trs", "tsr", "rts", "rst", "srt", "str"]]
p7 = [Primitive("rep{}".format(i), trep, j) for i, j in enumerate(range(7))]




primitives = p1 + p2 + p3 + p4 + p5 + p6 + p7



