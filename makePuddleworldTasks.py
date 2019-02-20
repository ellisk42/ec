"""
Makes Puddleworld tasks.
Tasks are (gridworld, text instruction) -> goal coordinate.
Credit: tasks are taken from: https://github.com/JannerM/spatial-reasoning 
"""

OBJECT_NAMES = ["NULL", "puddle", "star", "circle", "triangle", "heart", "spade", "diamond", "rock", "tree", "house", "horse"]

def loadPuddleWorldTasks(datafile='data/puddleworld/puddleworld.pickle'):
	"""
	Loads a pre-processed version of the Puddleworld tasks.
	"""
	import dill
	import pickle as pickle

	with open(path, "rb") as handle:
            result = dill.load(handle)
    return result

def makeLocalTasks():
	data = loadPuddleWorldTasks()
	return data['local_train'], data['local_test']

def makeGlobalTasks():
	data = loadPuddleWorldTasks()
	return data['global_train'], data['global_test']