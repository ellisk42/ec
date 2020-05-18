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

# USE_NEW_PRIMITIVES=True

class SupervisedDraw(Task):
	def __init__(self, name, program, USE_NEW_PRIMITIVES):
		# default should be USE_NEW_PRIMITIVES=True
		assert USE_NEW_PRIMITIVES in [True, False]

		if USE_NEW_PRIMITIVES:
			c = arrow(tstroke, tstroke)
		else:
			c = tstroke
		super(SupervisedDraw, self).__init__(name, c, [],
											  features=[]) # TODO: LT, needs this, i.e., a request.

		# compute the trajectory, which is a list of lines
		trajectory = []
		for segments in program:
			for i in range(segments.shape[0] - 1):
				p1 = segments[i][0], segments[i][1]
				p2 = segments[i+1][0], segments[i+1][1]
				trajectory.append([p1,p2])
		self.specialTask = ("draw",
							{"trajectory": trajectory,
                                                         "bounded_cost": True})

		self.strokes = program # list of np arrays.
		self.rendered_strokes = prog2pxl(program)

	def logLikelihood(self, e, timeout=None): # TODO, LT, given expression, calculates distance.
		# from dreamcoder.domains.tower.tower_common import centerTower
		assert False, "not using this I thought"
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

			print(l)
			if l>0.1:
				return NEGATIVEINFINITY
			else:
				return 0.0

	
def makeSupervisedTasks(trainset="S8full", doshaping="True", userealnames=True, USE_NEW_PRIMITIVES=False): # TODO, LT, make these tasks.
	# arches = [SupervisedTower("arch leg %d"%n,
	#                           "((for i %d v) (r 4) (for i %d v) (l 2) h)"%(n,n))
	#           for n in range(1,9)
	# ]

	assert USE_NEW_PRIMITIVES in [True, False]

	print("DRAW TASK training set: {}".format(trainset))
	print("DO SHAPING: {}".format(doshaping))
	# everything = arches + simpleLoops + Bridges + archesStacks + aqueducts + offsetArches + pyramids + bricks + staircase2 + staircase1 + compositions
	programs = []
	programnames = []
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


	# ===== train on basic stimuli like lines
	if doshaping:
		print("INCLUDING SHAPING STIMULI")
		ll = transform(_line, theta=pi/2, s=4, y=-2.)
		programs.extend([
			_line,
			transform(_circle, s=2.),
			transform(_circle, theta=pi/2),
			transform(_line, theta=pi/2),
			transform(_line, s=4),
			transform(_line, y=-2.),
			transform(_line, theta=pi/2, s=4.),
			transform(_line, theta=pi/2, y=-2.),
			transform(_line, theta=pi/2, s=4, y=-2.)]
			)
		programnames.extend(["shaping_{}".format(n) for n in range(9)])

	##############################################
	################# TRAINING SETS
	if trainset=="S8full":
		libname = "dreamcoder/domains/draw/trainprogs/S8_shaping"
		with open("{}.pkl".format(libname), 'rb') as fp:
			P = pickle.load(fp)
		programs.extend(P)
		assert 1==2 # note: have to add programnames

	if trainset in ["S8full", "S8"]:
		libname = "dreamcoder/domains/draw/trainprogs/S8"
		with open("{}.pkl".format(libname), 'rb') as fp:
			P = pickle.load(fp)
		programs.extend(P)
		assert 1==2 # note: have to add programnames

	if trainset=="S9full":
		libname = "dreamcoder/domains/draw/trainprogs/S9_shaping"
		with open("{}.pkl".format(libname), 'rb') as fp:
			P = pickle.load(fp)
		programs.extend(P)
		assert 1==2 # note: have to add programnames

	if trainset in ["S9full", "S9"]:
		libname = "dreamcoder/domains/draw/trainprogs/S9"
		with open("{}.pkl".format(libname), 'rb') as fp:
			P = pickle.load(fp)
		programs.extend(P)
		assert 1==2 # note: have to add programnames

	if trainset=="S8_nojitter":
		libname = "dreamcoder/domains/draw/trainprogs/S8_nojitter_shaping"
		with open("{}.pkl".format(libname), 'rb') as fp:
			P = pickle.load(fp)
		programs.extend(P)
		programnames.extend(["S8_nojitter_shaping_{}".format(n) for n in range(len(P))])

		libname = "dreamcoder/domains/draw/trainprogs/S8_nojitter"
		with open("{}.pkl".format(libname), 'rb') as fp:
			P = pickle.load(fp)
		programs.extend(P)
		programnames.extend(["S8_nojitter_{}".format(n) for n in range(len(P))])


	if trainset=="S9_nojitter":
		libname = "dreamcoder/domains/draw/trainprogs/S9_nojitter_shaping"
		with open("{}.pkl".format(libname), 'rb') as fp:
			P = pickle.load(fp)
		programs.extend(P)
		programnames.extend(["S9_nojitter_shaping_{}".format(n) for n in range(len(P))])

		libname = "dreamcoder/domains/draw/trainprogs/S9_nojitter"
		with open("{}.pkl".format(libname), 'rb') as fp:
			P = pickle.load(fp)
		programs.extend(P)
		programnames.extend(["S9_nojitter_{}".format(n) for n in range(len(P))])

	if trainset=="S10":
		libname = "dreamcoder/domains/draw/trainprogs/S10"
		with open("{}.pkl".format(libname), 'rb') as fp:
			P = pickle.load(fp)
		programs.extend(P)
		programnames.extend(["task{}".format(n) for n in range(len(P))])


	def addPrograms(lib, programs, programnames, nameprefix=[]):
		if not nameprefix:
			nameprefix=lib
			# note: assumes that name prefix is lib. here tell it otherwise.

		# ========= 1) SHAPING:
		# ---- get programs
		libname = "dreamcoder/domains/draw/trainprogs/{}".format(lib)
		with open("{}.pkl".format(libname), 'rb') as fp:
			P = pickle.load(fp)
		programs.extend(P)

		# ---- get program names
		with open("{}_stimnum.pkl".format(libname), 'rb') as fp:
			stimnum = pickle.load(fp)
		names = ["{}_{}".format(nameprefix, s) for s in stimnum]
		programnames.extend(names)

		return programs, programnames

	if trainset in ["S12", "S13"]:

		programs, programnames = addPrograms("S12_13_shaping", programs, programnames)
		programs, programnames = addPrograms(trainset, programs, programnames)


	# ===== make programs
	if userealnames:
		assert len(programs) == len(programnames)
		names = programnames
	else:
		names = ["task{}".format(i) for i in range(len(programs))]
	print("training task names:")
	print(names)
	alltasks = []
	for name, p in zip(names, programs):
	# for i, p in enumerate(programs):
		# name = "task{}".format(i)
		alltasks.append(SupervisedDraw(name, p, USE_NEW_PRIMITIVES=USE_NEW_PRIMITIVES))



	##############################################
	################# make test tasks?
	programs_test = []
	programs_test_names = []
	testtasks = []
	if trainset in ["S8full", "S8", "S9full", "S9", "S8_nojitter", "S9_nojitter"]:
		
		libname = "dreamcoder/domains/draw/trainprogs/S8_test"
		with open("{}.pkl".format(libname), 'rb') as fp:
			P = pickle.load(fp)
		programs_test.extend(P)	
		programs_test_names.extend(["S8_{}".format(n) for n in [0, 2, 59, 65, 94]])

		libname = "dreamcoder/domains/draw/trainprogs/S9_test"
		with open("{}.pkl".format(libname), 'rb') as fp:
			P = pickle.load(fp)
		programs_test.extend(P)	
		programs_test_names.extend(["S9_{}".format(n) for n in [14, 15, 17, 18, 29, 43, 55, 59, 61, 86, 96, 99, 140]])

		libname = "dreamcoder/domains/draw/trainprogs/S8_nojitter_test"
		with open("{}.pkl".format(libname), 'rb') as fp:
			P = pickle.load(fp)
		programs_test.extend(P)	
		programs_test_names.extend(["S8_nojitter_{}".format(n) for n in [69, 73, 134, 137, 139]])

		libname = "dreamcoder/domains/draw/trainprogs/S9_nojitter_test"
		with open("{}.pkl".format(libname), 'rb') as fp:
			P = pickle.load(fp)
		programs_test.extend(P)	
		programs_test_names.extend(["S9_nojitter_{}".format(n) for n in [56, 59, 76, 80, 108, 112, 135, 139, 144, 147]])

	if trainset in ["S12", "S13"]:
		programs_test, programs_test_names = addPrograms("S12_13_test", programs_test, programs_test_names)
		programs_test, programs_test_names = addPrograms("S12_test", programs_test, programs_test_names, nameprefix="S12")
		programs_test, programs_test_names = addPrograms("S13_test", programs_test, programs_test_names, nameprefix="S13")


	if programs_test:
		if userealnames:
			assert len(programs_test) == len(programs_test_names)
			names = programs_test_names
		else:
			names = ["test{}".format(i) for i in range(len(programs_test))]

		for name, p in zip(names, programs_test):
		# for i, p in enumerate(programs_test):
			# name = "test{}".format(i)
			testtasks.append(SupervisedDraw(name, p, USE_NEW_PRIMITIVES=USE_NEW_PRIMITIVES))
	print("test tasks:")
	print(names)
	
	return alltasks, testtasks, programnames, programs_test_names

