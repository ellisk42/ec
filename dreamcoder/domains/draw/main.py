from dreamcoder.domains.draw.makeDrawTasks import drawDrawings
from dreamcoder.domains.draw.drawPrimitives import primitives, taxes, tartist, tangle, tscale, tdist

# from dreamcoder.dreamcoder import ecIterator
from dreamcoder.grammar import Grammar
# from dreamcoder.program import Program
# from dreamcoder.recognition import variable, maybe_cuda
from dreamcoder.task import Task
from dreamcoder.type import arrow
# from dreamcoder.utilities import eprint, testTrainSplit, loadPickle

def dreamFromGrammar(g, directory, N=50):
	request = arrow(taxes, taxes)
	programs = [ p for _ in range(N) for p in [g.sample(request, maximumDepth=20)] if p is not None]
	drawDrawings(*programs, filenames)