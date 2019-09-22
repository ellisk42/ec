from dreamcoder.program import Primitive, Program
from dreamcoder.utilities import Curried
from dreamcoder.grammar import Grammar
from dreamcoder.type import baseType, arrow

from matplotlib import patches
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from matplotlib import transforms
import matplotlib
import math
import numpy as np


NBINS = 10 # for discretizing continious params
XYLIM = 5. # size of canvas.

# ========== base types
tdummy = baseType("tdummy")
taxes = baseType("taxes")
tartist = baseType("tartist")
tangle = baseType("tangle")
tscale = baseType("tscale")
tdist = baseType("tdist")
ttrorder = baseType("ttorder")

# -- for repeats
tartistfun = baseType("tartistfun")
ttransform = baseType("ttransform")
trep = baseType("trep")
# tintrep = baseType("tintrep") # number for repeat

# ========== Primitives

# def _makeBlankAxes():

def _blankAxes():
	# initialize canvas
	legacyFormat=False
	if legacyFormat:
		fig = plt.figure(edgecolor="w")
		ax = plt.axes()
		ax.axis("off")
		#     center = [0,0]
		plt.xlim([-XYLIM, XYLIM])
		plt.ylim([-XYLIM, XYLIM])
		ax.set_aspect("equal")
	else:
		# new version, good for making pixels, centered, etc.
		fig = plt.figure(figsize=(3,3), edgecolor="w")
		ax = fig.add_axes([-0.03, -0.03, 1.06, 1.06]) # outside bounds, since edge is sometimes weird.
		ax.axis("off")
		ax.set_xlim(-XYLIM, XYLIM)
		ax.set_ylim(-XYLIM, XYLIM)
	return ax

def _line(dummy=0):
	x, y, l = (0, 0, 1)
	l = patches.FancyArrow(x, y, l, 0, length_includes_head=True,
		head_width=0, head_length=0, fc='k', ec='k')
	return l

# def _makeline():
# 	x, y, l = (0, 0, 1)
# 	l = patches.FancyArrow(x, y, l, 0, length_includes_head=True,
# 		head_width=0, head_length=0, fc='k', ec='k')
# 	return l

# _line = makeline()

# x, y, l = (0, 0, 1)
# _line = patches.FancyArrow(x, y, l, 0, length_includes_head=True,
# 	head_width=0, head_length=0, fc='k', ec='k')


def _circle(dummy=0):
	def C():
		x, y, r = (0, 0, 1)
		c = patches.Circle((x, y), radius=r, color='white', ec='black')
		return c
	return _transform(C(), s=0.5)

# x, y, r = (0, 0, 1)
# _circle = patches.Circle((x, y), radius=r, color='white', ec='black')

def _arc(dummy=0):
	# TODO: ie., a part of a circle
	pass


def _ellipse():
    "or this could be derived from four arcs?"
    pass

def _square():
    " or this could be derived from four lines"
    pass


def _connect(p1, p2):
	"takes in two primitives and outputs a primitive that is a list of the first two. recursive."
	p = []
	
	if isinstance(p1, list):
		p.extend(p1)
	else:
		p.append(p1)

	if isinstance(p2, list):
		p.extend(p2)
	else:
		p.append(p2)

	return p


def _tform(p, s, theta, x, y, order):
	# order is one of the 6 ways you can permutate the three transformation primitives. 
	# write as a string (e.g. "trs" means scale, then rotate, then tranlate.)
	# input and output types guarantees a primitive will only be transformed once.

	def _translate(p, x, y):
		# p is either a primitive or a collection of primitives
		t1 = p.get_transform()
		tt = transforms.Affine2D().translate(x,y)
		t = t1 + tt
		p.set_transform(t)
		return p

	def _rotate(p, theta):
		# p is either a primitive or a collection of primitives. center is cneter of rotation.
		center = [0,0]
		t1 = p.get_transform()
		tr = transforms.Affine2D().rotate_around(center[0], center[1], theta)
		t = t1 + tr
		p.set_transform(t)
		return p

	def _scale(p, s):
		# p is either a primitive or a collection of primitives. center is cneter of rotation.
		# s is scale ratio (1 is identity)
		t1 = p.get_transform()
		ts = transforms.Affine2D().scale(s)
		t = t1 + ts
		p.set_transform(t)
		return p  

	if order == "trs":
		return _translate(_rotate(_scale(p, s), theta), x, y)
	elif order == "tsr":
		return _translate(_scale(_rotate(p, theta), s), x, y)
	elif order == "rts":
		return _rotate(_translate(_scale(p, s), x, y), theta)
	elif order == "rst":
		return _rotate(_scale(_translate(p, x, y), s), theta)
	elif order == "srt":
		return _scale(_rotate(_translate(p, x, y), theta), s)
	elif order == "str":
		return _scale(_translate(_rotate(p, theta), x, y), s)

def _transform(p, s=1, theta=0, x=0, y=0, order="trs"):
    if isinstance(p, list):
        pout = [_tform(pp, s, theta, x, y, order) for pp in p]
    else:
        pout = _tform(p, s, theta, x, y, order)
    return pout


def _reflect_y(p):
	# reflects across y axis. call reflect() instead.
	t1 = p.get_transform()
	tf = transforms.Affine2D().scale(-1, 1)
	t = t1 + tf
	p.set_transform(t)
	return p

def _reflect(p, theta=math.pi/2):
	# first rotate p by -theta, then reflect across y axis, then unrotate (by +theta)
	# y axis would be theta = pi/2
	th = theta - math.pi/2
	
	def _A(p, th):
		p = _transform(p, theta=-th)
		p = _reflect_y(p)
		p = _transform(p, theta=th)
		return p

	if isinstance(p, list):
		pout = [_A(pp, th) for pp in p]
	else:
		pout = _A(p, th)
	return pout

def _draw(ax, P):

	def c(p):
		# where p is list of primitives
		# outputs a collecction of primitives. essentially takes union. can treat this as a new primitive
		coll = PatchCollection(p)
		coll.set_edgecolor('k')
		coll.set_facecolor([0,0,0,0])
		#     coll.set_alpha(0.5)
		return coll

	def _D(ax, P):
		P = c([P]) # first convert to patch collection.
		ax.add_collection(P) # second, add to axis.
		return ax

	if isinstance(P, list):
		for pp in P:
			ax = _D(ax, pp)
	else:
		ax = _D(ax, P)
	return ax


# --- function versions of line, circle, and Transform, required as a type, for _transform.
def _Tfun(s=1, theta=0, x=0, y=0, order="trs"):
	t = lambda p: _transform(p, s, theta, x, y, order)
	return t

def _circlefun(dummy=0):
	c = lambda : _circle(dummy)
	return c

def _linefun(dummy=0):
	l = lambda : _line(dummy)
	return l

def _repeat(ax, P, N, T):
# UPDATED VERSION OF REPEAT (here takes in a T and outputs a list (without connect)
# outputs list of primitives, each transformed 1, 2, ..., N times
# p = lambda: line()
# T = lambda p: transform(p, theta=pi/2)
	for i in range(N):
	#         Pthis = copy.deepcopy(P)
	#         Pthis = matplotlib.artist.Artist()
	#         Pthis.update_from(P)
	#         Pthis = P
		Pthis = P()
		for _ in range(i):
			Pthis = T(Pthis)
		_draw(ax, Pthis)
	return ax


# ----- for generating figures
def _save(ax, fname):
	plt.savefig(fname, "png")

def _drawNsave(prog, libname, stimname):
    dirname_svg = "{}/svg/{}".format(SAVEDIR, libname)
    dirname_png = "{}/png/{}".format(SAVEDIR, libname)    
    
    if not os.path.exists(dirname_svg):
        os.makedirs(dirname_svg)    # prog is artist or collection of artists, or list of those
    if not os.path.exists(dirname_png):
        os.makedirs(dirname_png)    # prog is artist or collection of artists, or list of those
        
    ax = _draw(blankAxes(), prog)
    ax.get_figure().savefig("{}/{}.svg".format(dirname_svg, stimname))
    ax.get_figure().savefig("{}/{}.png".format(dirname_png, stimname))
#     ax.get_figure().close()

# blankaxes = Primitive("blankaxes", taxes, Curried(_blankAxes))
# line = Primitive("line", tartist, Curried(_line))
# circle = Primitive("circle", tartist, Curried(_circle))
# scale = Primitive("scale", arrow(tartist, tscale, tartist), Curried(_scale))
# rotate = Primitive("rotate", arrow(tartist, tangle, tartist), Curried(_rotate))
# translate = Primitive("translate", arrow(tartist, tdist, tdist, tartist), 
# 	Curried(_translate))
# reflect = Primitive("reflect", arrow(tartist, tangle, tartist), Curried(_reflect))
# draw = Primitive("draw", arrow(taxes, tartist, taxes), Curried(_draw))

# Primitive("scale", arrow(tartist, tscale, tartist), Curried(_scale)), 
# Primitive("rotate", arrow(tartist, tangle, tartist), Curried(_rotate)), 
# Primitive("translate", arrow(tartist, tdist, tdist, tartist), 
# 	Curried(_translate)), 


p1 = [
	# Primitive("blankaxes", taxes, _blankAxes), 
	# Primitive("line", tartist, _line), 
	# Primitive("circle", tartist, _circle),
	Primitive("line", arrow(tdummy, tartist), Curried(_line)), 
	Primitive("circle", arrow(tdummy, tartist), Curried(_circle)),
	Primitive("transform", arrow(tartist, tscale, tangle, tdist, tdist, ttrorder, tartist), Curried(_transform)),
	Primitive("reflect", arrow(tartist, tangle, tartist), Curried(_reflect)), 
	Primitive("draw", arrow(taxes, tartist, taxes), Curried(_draw)),
	Primitive("dummy", tdummy, 0)
]
p2 = [Primitive("scale{}".format(i), tscale, 2.0**j) for i, j in enumerate(np.linspace(-1.0, 1.0, NBINS))]
p3 = [Primitive("dist{}".format(i), tdist, j) for i, j in enumerate(np.linspace(-4, 4, 9))]
p4 = [Primitive("angle{}".format(i), tangle, j*math.pi/4) for i, j in enumerate(range(8))]
p5 = [Primitive(j, ttrorder, j) for j in ["trs", "tsr", "rts", "rst", "srt", "str"]]

if True:
	tartistfun = baseType("tartistfun")
	ttransform = baseType("ttransform")
	trep = baseType("trep")

	p6 = [
		Primitive("circlefun", arrow(tdummy, tartistfun), Curried(_circlefun)),
		Primitive("linefun", arrow(tdummy, tartistfun), Curried(_linefun)),
		Primitive("Tfun", arrow(tscale, tangle, tdist, tdist, ttrorder, ttransform), Curried(_Tfun)),
		Primitive("rep", arrow(taxes, tartistfun, trep, ttransform, taxes), Curried(_repeat))]
	p7 = [Primitive("rep{}".format(i), trep, j) for i, j in enumerate(range(8))]

primitives = p1 + p2 + p3 + p4 + p5 + p6 + p7


# def _repeat(ax, P, N, dx, dy, dtheta):
#     # rotfirst = 0,1 --> if 1, then first rotates before translating, else first translates
#     # P must be function that generates primtiive or collection of prim (e.g. lambda fucntion)
#     rotfirst = 1

#     for i in range(N):
# #         Pthis = copy.deepcopy(P)
# #         Pthis = matplotlib.artist.Artist()
# #         Pthis.update_from(P)
# #         Pthis = P
#         Pthis = P()
#         for _ in range(i):
#             if rotfirst==1:
#                 Pthis = rotate(Pthis, dtheta)
#                 Pthis = translate(Pthis, dx, dy)
#             else:
#                 Pthis = translate(Pthis, dx, dy)
#                 Pthis = rotate(Pthis, dtheta)
#         draw(ax, Pthis)
#     return ax

# Primitive("repeat", arrow(taxes, tartist, tintrep, tdist, 
# 	tdist, tangle, taxes), Curried(_repeat))

