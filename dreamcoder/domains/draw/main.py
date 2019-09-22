# from dreamcoder.domains.draw.makeDrawTasks import drawDrawings
from dreamcoder.domains.draw.drawPrimitives import primitives, taxes, tartist, tangle, tscale, tdist

# from dreamcoder.dreamcoder import ecIterator
from dreamcoder.grammar import Grammar
# from dreamcoder.program import Program
# from dreamcoder.recognition import variable, maybe_cuda
from dreamcoder.task import Task
from dreamcoder.type import arrow
# from dreamcoder.utilities import eprint, testTrainSplit, loadPickle

g0 = Grammar.uniform(primitives)

def dreamFromGrammar(g=g0, directory = "", N=50):
	# request = taxes # arrow9turtle turtle) just for logl.
	request = arrow(taxes, taxes) # arrow9turtle turtle) just for logl.
	programs = [ p for _ in range(N) for p in [g.sample(request, maximumDepth=20)] if p is not None]
	return programs
	# drawDrawings(*programs, filenames)

