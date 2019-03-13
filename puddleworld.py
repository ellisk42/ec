"""
Puddleworld.
Tasks are (gridworld, text instruction) -> goal coordinate.
Credit: https://github.com/JannerM/spatial-reasoning 
"""
from ec import explorationCompression, commandlineArguments, Task, ecIterator
from makePuddleworldTasks import makeLocalTasks, makeGlobalTasks
from utilities import eprint, numberOfCPUs

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
	args = commandlineArguments(
		enumerationTimeout=10, activation='tanh', iterations=10, recognitionTimeout=3600,
		a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
		helmholtzRatio=0.5, structurePenalty=1.,
		CPUs=numberOfCPUs(),
		extras=puddleworld_options)

	doLocal, doGlobal = args.pop('local'), args.pop('global')
	eprint("Using local tasks: %r, Using global tasks: %r" % (doLocal, doGlobal))

	localTrain, localTest = makeLocalTasks() if doLocal else []
	globalTrain, globalTest = makeGlobalTasks() if doGlobal else []
		

	assert False