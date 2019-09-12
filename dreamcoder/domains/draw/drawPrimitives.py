from dreamcoder.program import Primitive, Program
from dreamcoder.utilities import Curried
from dreamcoder.type import Grammar, baseType

from matplotlib import patches
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from matplotlib import transforms
import matplotlib
import math
import numpy as np


NBINS = 10 # for discretizing continious params
XYLIM = 5 # size of canvas.

# ========== base types
taxes = baseType("axes")
tartist = baseType("tartist")
tangle = baseType("tangle")
tscale = baseType("tscale")
tdist = baseType("tdist")
# tintrep = baseType("tintrep") # number for repeat

# ========== Primitives
def _blankAxes():
    # initialize canvas
    fig = plt.figure(edgecolor='w')
    ax = plt.axes()
#     center = [0,0]
    plt.xlim([-XYLIM, XYLIM])
    plt.ylim([-XYLIM, XYLIM])
    ax.set_aspect("equal")
    return ax

def _line():
    x, y, l = (0, 0, 1)
    l = patches.FancyArrow(x, y, l, 0, length_includes_head=True,
                     head_width=0, head_length=0, fc='k', ec='k')
    return l

def _circle():
    x, y, r = (0, 0, 1)
    c = patches.Circle((x, y), radius=r, color='white', ec='black')
    return c

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
    theta = theta - math.pi/2
    p = _rotate(p, -theta)
    p = _reflect_y(p)
    p = _rotate(p, theta)
    return p

def _draw(ax, P):

	def c(p):
	    # where p is list of primitives
	    # outputs a collecction of primitives. essentially takes union. can treat this as a new primitive
	    coll = PatchCollection(p)
	    coll.set_edgecolor('k')
	    coll.set_facecolor([0,0,0,0])
	#     coll.set_alpha(0.5)
	    return coll
    P = c([P])
    ax.add_collection(P)
    return ax

# blankaxes = Primitive("blankaxes", taxes, Curried(_blankAxes))
# line = Primitive("line", tartist, Curried(_line))
# circle = Primitive("circle", tartist, Curried(_circle))
# scale = Primitive("scale", arrow(tartist, tscale, tartist), Curried(_scale))
# rotate = Primitive("rotate", arrow(tartist, tangle, tartist), Curried(_rotate))
# translate = Primitive("translate", arrow(tartist, tdist, tdist, tartist), 
# 	Curried(_translate))
# reflect = Primitive("reflect", arrow(tartist, tangle, tartist), Curried(_reflect))
# draw = Primitive("draw", arrow(taxes, tartist, taxes), Curried(_draw))



primitives = [
	Primitive("blankaxes", taxes, Curried(_blankAxes))
	Primitive("line", tartist, Curried(_line))
	Primitive("circle", tartist, Curried(_circle))
	Primitive("scale", arrow(tartist, tscale, tartist), Curried(_scale))
	Primitive("rotate", arrow(tartist, tangle, tartist), Curried(_rotate))
	Primitive("translate", arrow(tartist, tdist, tdist, tartist), 
		Curried(_translate))
	Primitive("reflect", arrow(tartist, tangle, tartist), Curried(_reflect))
	Primitive("draw", arrow(taxes, tartist, taxes), Curried(_draw))
] + [Primitive("scale{}".format(j), tscale, 2.0**j) for j in np.linspace(-1.0, 1.0, NBINS)]
+ [Primitive("angle{}".format(j), tangle, j*math.pi/4) for j in range(8)]
+ [Primitive("dist{}".format(), tdist, j) for j in np.linspace(-4, 4, 9)]

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

