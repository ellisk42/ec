from dreamcoder.domains.draw.drawPrimitives import *
from dreamcoder.domains.draw.drawPrimitives import _tform, _line, _circle, _repeat, _makeAffine, _reflect, _connect
# from dreamcoder.dreamcoder import ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
# from dreamcoder.recognition import variable, maybe_cuda
from dreamcoder.task import Task
from dreamcoder.type import arrow
from dreamcoder.utilities import *
import math
from math import pi

import pickle
import numpy as np


class SupervisedDraw(Task):
	def __init__(self, name, program):
		super(SupervisedDraw, self).__init__(name, tstroke, [],
											  features=[]) # TODO: LT, needs this, i.e., a request.

		# compute the trajectory, which is a list of lines
		trajectory = []
		for segments in program:
			for i in range(segments.shape[0] - 1):
				p1 = segments[i][0], segments[i][1]
				p2 = segments[i+1][0], segments[i+1][1]
				trajectory.append([p1,p2])
		self.specialTask = ("draw",
							{"trajectory": trajectory})

		self.strokes = program # list of np arrays.
		self.rendered_strokes = prog2pxl(program)

	def logLikelihood(self, e, timeout=None): # TODO, LT, given expression, calculates distance.
		# from dreamcoder.domains.tower.tower_common import centerTower
		if False:
			p1 = self.rendered_strokes
			p2 = fig2pixel(e.evaluate([]))

			# l = loss(p1, p2, smoothing=2) 
			l = loss_pxl(p1, p2)

			if l>0.1:
				return NEGATIVEINFINITY
			else:
				return 0.0
		else:
			# print("doing it!")
			# p1 = 
			# p2 = prog2pxl(e.evaluate([]))

			if not hasattr(e, "rendering"):
				e.rendering = prog2pxl(e.evaluate([]))

			# l = loss(p1, p2, smoothing=2) 
			l = loss_pxl(self.rendered_strokes, e.rendering)

			if l>0.1:
				return NEGATIVEINFINITY
			else:
				return 0.0

	
def makeSupervisedTasks(): # TODO, LT, make these tasks.
	# arches = [SupervisedTower("arch leg %d"%n,
	#                           "((for i %d v) (r 4) (for i %d v) (l 2) h)"%(n,n))
	#           for n in range(1,9)
	# ]
					 
	# everything = arches + simpleLoops + Bridges + archesStacks + aqueducts + offsetArches + pyramids + bricks + staircase2 + staircase1 + compositions
	alltasks = []
	programs = []

	if False:
		# to test each primitive and rule
		programs = [
		_tform(_line, _makeAffine(x=0.25)),
		_line + _circle + _tform(_circle, _makeAffine(x=1.0)),
		_tform(_line, _makeAffine(x=1.0)),
		_tform(_line, _makeAffine(x=1.0, theta=pi/2)),
		_tform(_line, _makeAffine(x=1.0, s=2.0)),
		_tform(_line, _makeAffine(x=1.0, s=0.5, order="rst")),
		_repeat(_line, 2, _makeAffine(x=1.0)),
		_repeat(_connect(_tform(_circle, _makeAffine(s=1.5)), _line), 2, _makeAffine(theta=pi/2))		
		]

	if False:
		programs = [_line + _circle,
		_tform(_line, _makeAffine(x=1.0)),
		_tform(_line, _makeAffine(x=1.0, theta=pi/2)),
		_tform(_line, _makeAffine(x=1.5)),
		_tform(_line, _makeAffine(x=0.5)),
		_tform(_line, _makeAffine(x=2.0)),
		_tform(_circle, _makeAffine(x=1.5)),
		_repeat(_line, 2, _makeAffine(x=1.0)),
		_tform(_line, _makeAffine(x=2.)) + _tform(_circle, _makeAffine(x=1.)),
		_line + _tform(_line, _makeAffine(x=2.)) + _tform(_circle, _makeAffine(x=1.)),
		_repeat(_line+_tform(_circle, _makeAffine(x=1.)), 3, _makeAffine(theta=math.pi/2)),
		]

	# -- add some programs used in behaivor
	if False:
		libname = "dreamcoder/domains/draw/S6"
		with open("{}.pkl".format(libname), 'rb') as fp:
			P = pickle.load(fp)
		programs.extend(P[:50])

	if True:
		ll = transform(_line, theta=pi/2, s=4, y=-2.),
		programs.extend([
			_line,
			ll,
			transform(_line, s=4),
			transform(_line, y=-2.),
			transform(_line, theta=pi/2),
			transform(_line, theta=pi/2, y=-2.),
			transform(_line, theta=pi/2, s=4, y=-2.)]			
			)
		libname = "dreamcoder/domains/draw/trainprogs/S8_shaping"
		with open("{}.pkl".format(libname), 'rb') as fp:
			P = pickle.load(fp)
		programs.extend(P)

	if True:
		libname = "dreamcoder/domains/draw/trainprogs/S8"
		with open("{}.pkl".format(libname), 'rb') as fp:
			P = pickle.load(fp)
		programs.extend(P)

	for i, p in enumerate(programs):
		name = "task{}".format(i)
		alltasks.append(SupervisedDraw(name, p))

	return alltasks

