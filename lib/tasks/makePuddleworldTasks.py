"""
Makes Puddleworld tasks.
Tasks are (gridworld, text instruction) -> goal coordinate.
Credit: tasks are taken from: https://github.com/JannerM/spatial-reasoning 
"""
from puddleworldPrimitives import *
from lib.tasks.task import *
from type import *

OBJECT_NAMES = ["NULL", "puddle", "star", "circle", "triangle", "heart", "spade", "diamond", "rock", "tree", "house", "horse"]

def loadPuddleWorldTasks(datafile='data/puddleworld/puddleworld.json'):
	"""
	Loads a pre-processed version of the Puddleworld tasks.
	"""
	import json

	with open(datafile) as f:
		result = json.load(f)
	return result

def makePuddleworldTask(raw_task):
	"""
	Converts a raw task with 
	layouts (NxN array), 
	Objects (NxN array of object locations), 
	Instructions (string) and 
	Goals ((X, Y) coordinate)
	into a task.
	"""
	layouts, objects, instructions, goals = raw_task 
	task = Task(name=instructions, 
				request=(arrow(tpair(tLayoutMap, tObjectMap), tLocation)),
				examples=[((layouts, objects), goals)],
				features=instructions)
	return task


def makeTasks(train_key, test_key):
	data = loadPuddleWorldTasks()
	raw_train, raw_test = data[train_key], data[test_key]

	train, test = [makePuddleworldTask(task) for task in raw_train], [makePuddleworldTask(task) for task in raw_test]

	print(train[0].name)
	print(train[0].examples)
	print(train[0].features)
	return train, test

def makeLocalTasks():
	return makeTasks('local_train', 'local_test')

def makeGlobalTasks():
	return makeTasks('global_train', 'global_test')
