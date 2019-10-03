# from dreamcoder.domains.draw.makeDrawTasks import drawDrawings
from dreamcoder.domains.draw.drawPrimitives import *
from dreamcoder.domains.draw.makeDrawTasks import makeSupervisedTasks, SupervisedDraw
from dreamcoder.dreamcoder import ecIterator
from dreamcoder.grammar import Grammar
# from dreamcoder.program import Program
# from dreamcoder.recognition import variable, maybe_cuda
from dreamcoder.task import Task
from dreamcoder.type import arrow
from dreamcoder.recognition import ImageFeatureExtractor
# from dreamcoder.utilities import eprint, testTrainSplit, loadPickle
import datetime
import os

class DrawCNN(ImageFeatureExtractor):
        special = "draw"
        def __init__(self, tasks, testingTasks=[], cuda=False):
                super(DrawCNN, self).__init__(inputImageDimension=128,
                                              resizedDimension=64,
                                              cuda=cuda,
                                              channels=1)
                print("output dimensionality",self.outputDimensionality)
        def taskOfProgram(self, p, t):
                return SupervisedDraw("dream", p.evaluate([]))

        def featuresOfTask(self, t):
                return self(t.rendered_strokes)

g0 = Grammar.uniform(primitives)

def dreamFromGrammar(g=g0, directory = "", N=25):
	# request = taxes # arrow9turtle turtle) just for logl.
	# request = arrow(taxes, taxes) # arrow9turtle turtle) just for logl.
	request = tstroke # arrow9turtle turtle) just for logl.
	programs = [ p for _ in range(N) for p in [g.sample(request, maximumDepth=15)] if p is not None]
	return programs
	# drawDrawings(*programs, filenames)

def main_dummy(N=25):
        ps = dreamFromGrammar(N=N)
        for n,p in enumerate(ps):
                print(n,p)
                a = p.evaluate([])
                savefig(a, f"/tmp/draw{n}.png")
                # a.savefig(f"/tmp/draw{n}.png")


def main(arguments):
	g0 = Grammar.uniform(primitives)

	train = makeSupervisedTasks()

	timestamp = datetime.datetime.now().isoformat()
	outputDirectory = "experimentOutputs/draw/%s"%timestamp
	evaluationTimeout = 0.001 # seconds, how long allowed

	os.system(f"mkdir -p {outputDirectory}")
	arguments["featureExtractor"] = DrawCNN
	generator = ecIterator(g0, train,
                               outputPrefix="%s/draw"%outputDirectory,
			       evaluationTimeout=evaluationTimeout,
                               **arguments) # 

	for result in generator:
		continue
