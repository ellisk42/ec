"""
Puddleworld.
Tasks are (gridworld, text instruction) -> goal coordinate.
Credit: https://github.com/JannerM/spatial-reasoning 
"""
from ec import explorationCompression, commandlineArguments, Task, ecIterator
from makeTextTasks import makeLocalTasks, makeGlobalTasks

def puddleworld_options(parser):
	parser.add_argument(
		"--local",
		action="store_true",
		default=True,
		help='Include local navigation tasks.'
		)
	parser.add_argument(
		"--global",
		action="store_true",
		default=False,
		help='Include global navigation tasks.'
		)

if __name__ == "__main__":
	arguments = commandlineArguments(
        recognitionTimeout=7200,
        iterations=10,
        helmholtzRatio=0.5,
        topK=2,
        maximumFrontier=5,
        structurePenalty=10.,
        a=3,
        activation="tanh",
        CPUs=numberOfCPUs(),
        featureExtractor=None, #TODO(cathywong)
        pseudoCounts=30.0,
        extras=text_options)

		doLocal, doGlobal = arguments.pop('local'), arguments.pop('global')
		eprint("Using local tasks: %r, Using global tasks: %r")

		localTrain, localTest = makeLocalTasks() if doLocal else []
		globalTrain, globalTest = makeGlobalTasks() if doGlobal else []
		

