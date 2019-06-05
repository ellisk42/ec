"""
Puddleworld.
Tasks are (gridworld, text instruction) -> goal coordinate.
Credit: https://github.com/JannerM/spatial-reasoning 
"""
import datetime
import os
import random

from lib.ec import commandlineArguments
from grammar import Grammar
from utilities import eprint, numberOfCPUs

from lib.domains.puddleworld.puddleworldPrimitives import primitives
from lib.domains.puddleworld.makePuddleworldTasks import makeLocalTasks, makeGlobalTasks


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
	parser.add_argument("--random-seed", 
		type=int, 
		default=0
		)

if __name__ == "__main__":
	args = commandlineArguments(
		enumerationTimeout=10, activation='tanh', iterations=10, recognitionTimeout=3600,
		a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
		helmholtzRatio=0.5, structurePenalty=1.,
		CPUs=numberOfCPUs(),
		extras=puddleworld_options)

	# Set up.
	random.seed(args.pop("random_seed"))
	timestamp = datetime.datetime.now().isoformat()
	outputDirectory = "experimentOutputs/puddleworld/%s"%timestamp
	os.system("mkdir -p %s"%outputDirectory)


	# Make tasks.
	doLocal, doGlobal = args.pop('local'), args.pop('global')
	localTrain, localTest = makeLocalTasks() if doLocal else [], []
	globalTrain, globalTest = makeGlobalTasks() if doGlobal else [], []
	eprint("Using local tasks: %d train, %d test" % (len(localTrain), len(localTest)))
	eprint("Using global tasks: %d train, %d test" % (len(globalTrain), len(globalTest)))
		
	# Make starting grammar.
	baseGrammar = Grammar.uniform(primitives)

	# Train.
	assert False