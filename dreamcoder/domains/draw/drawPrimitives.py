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
from dreamcoder.type import baseType, arrow

from primitives import *
from primitives import _makeAffine, _tform, _reflect, _repeat, _connect

matplotlib.use('TkAgg')

# ======= DEFINE ALL PRIMITIVES
tstroke = baseType("tstroke")
tangle = baseType("tangle")
tscale = baseType("tscale")
tdist = baseType("tdist")
ttrorder = baseType("ttorder")
ttransmat = baseType("ttransmat")
trep = baseType("trep")

p1 = [
	Primitive("line", tstroke, _line), 
	Primitive("circle", tstroke, _circle),
	Primitive("transmat", arrow(tscale, tangle, tdist, tdist, ttrorder, ttransmat), Curried(_makeAffine)),
	Primitive("transform", arrow(tstroke, ttransmat, tstroke), Curried(_tform)),
	Primitive("reflect", arrow(tstroke, tangle, tstroke), Curried(_reflect)), 
	Primitive("connect", arrow(tstroke, tstroke, tstroke), Curried(_connect)),
	Primitive("repeat", arrow(tstroke, trep, ttransmat, tstroke), Curried(_repeat))
]
p2 = [Primitive("scale{}".format(i), tscale, j) for i, j in enumerate(np.linspace(1.0, 4.0, 7))]
p3 = [Primitive("dist{}".format(i), tdist, j) for i, j in enumerate(np.linspace(-4, 4, 9))]
NANGLE = 8
p4 = [Primitive("angle{}".format(i), tangle, j*2*math.pi/NANGLE) for i, j in enumerate(range(NANGLE))]
p5 = [Primitive("angle{}".format(i), tangle, (j+1)*2*math.pi/3) for j,i in enumerate(range(NANGLE, NANGLE+2))]
p6 = [Primitive(j, ttrorder, j) for j in ["trs", "tsr", "rts", "rst", "srt", "str"]]
p7 = [Primitive("rep{}".format(i), trep, j) for i, j in enumerate(range(7))]

primitives = p1 + p2 + p3 + p4 + p5 + p6 + p7



