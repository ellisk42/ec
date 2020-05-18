""" A version not using continuation"""

# =========== [NEWER VERSION, NOT USING MATPLOTLIB]
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
from dreamcoder.domains.draw.primitives import _makeAffine, _tform, _reflect, _repeat, _connect, _line, _circle, _tform_wrapper, _reflect_wrapper, _emptystroke
from dreamcoder.domains.draw.primitives import _lineC, _circleC, _finishC, _repeatC, _transformC, _reflectC, _emptystrokeC

from dreamcoder.domains.draw.drawPrimitives import *


primitives = primitiveList(USE_NEW_PRIMITIVES=False) + getNewPrimitives()