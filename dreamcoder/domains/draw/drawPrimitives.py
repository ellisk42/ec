# =========== [NEWER VERSION, NOT USING MATPLOTLIB]
print("HERE")
import math
import numpy as np
# import matplotlibplt

from scipy.ndimage import gaussian_filter as gf
from skimage import color
from scipy.ndimage import gaussian_filter as gf
import cairo

from dreamcoder.program import Primitive, Program
from dreamcoder.utilities import Curried
from dreamcoder.grammar import Grammar
from dreamcoder.type import baseType, arrow, tmaybe, t0, t1, t2

from dreamcoder.domains.draw.primitives import *
from dreamcoder.domains.draw.primitives import _makeAffine, _tform, _reflect, _repeat, _connect, _line, _circle, _tform_wrapper, _reflect_wrapper

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

p0 = [Primitive("None", tmaybe(t0), None), Primitive("Some", arrow(t0, tmaybe(t0)), _givemeback)]


p1 = [
	Primitive("line", tstroke, _line), 
	Primitive("circle", tstroke, _circle),
	Primitive("transmat", arrow(tmaybe(tscale), tmaybe(tangle), tmaybe(tdist), tmaybe(tdist), tmaybe(ttrorder), ttransmat), Curried(_makeAffine)),
	Primitive("transform", arrow(tstroke, ttransmat, tstroke), Curried(_tform_wrapper)),
	Primitive("reflect", arrow(tstroke, tangle, tstroke), Curried(_reflect_wrapper)), 
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
p6 = [Primitive(j, ttrorder, j) for j in ORDERS]
p7 = [Primitive("rep{}".format(i), trep, j+1) for i, j in enumerate(range(7))]




primitives = p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7

def getPrimitives(trainset="", prune=False, primitives=primitives, fullpruning=True):
	if prune:
		assert len(trainset)>0, "Have to tell me which trainset to use for primtives"
		# -- then only keep a subset of primtiives
		if trainset in ["S12", "S13"]:

			print("Full primitives:")
			for p in primitives:
				print("{} = {}".format(p.name, p.evaluate([])))

			# ------- list of primtiives to remove
			primitives_to_remove = [] + \
			["scale{}".format(i) for i in [0, 1, 2, 3, 4, 5, 6]] + \
			["dist{}".format(i) for i in [0, 2, 5, 7, 10, 12, 13, 19, 20, 21, 22]] + \
			["angle{}".format(i) for i in [1,3,5,7,8,9]] + \
			["tsr", "srt", "str", "rts", "rst"] + \
			["rep{}".format(i) for i in [4,5,6]]
			
			if fullpruning:
				# then really careful remove anything not useful
				# partly motivated by seeing what DC actually uses given the partial pruning above.
				primitives_to_remove.extend(["dist1", "dist11", "dist8", "reflect", "angle4", "angle6"])

			print("removing these primitives:")
			print(primitives_to_remove)

			# ----- do removal
			primitives = [p for p in primitives if p.name not in primitives_to_remove]

			print("Primtives, after pruning:")
			for p in primitives:
				print("{} = {}".format(p.name, p.evaluate([])))

			return primitives
		else:
			print("DO NOT KNOW HOW TO PRUNE PRIMITIVES FOR THIS TRAINSET")
			raise
	else:
		return primitives

