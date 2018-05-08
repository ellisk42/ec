from type import tpregex
from task import Task
import pickle
import json


taskfile = './data_filtered.json'
#task_list = pickle.load(open(taskfile, 'rb'))


with open ('./data_filtered.json') as f:
	file_contents = f.read()
task_list = json.loads(file_contents)


def makeTasks():
	#a series of tasks

	#if I were to just dump all of them:
	regextasks = [ 
		Task("Luke data column no." + str(i), 
			tpregex, 
			[((), example) for example in task_list[i]]
			) for i in range(len(task_list))]
	"""	regextasks = [
       	Task("length bool", arrow(none,tstr),
             [((l,), len(l))
              for _ in range(10)
              for l in [[flip() for _ in range(randint(0,10)) ]] ]),
        Task("length int", arrow(none,tstr),
             [((l,), len(l))
              for _ in range(10)
              for l in [randomList()] ]),
    ]
	"""
	return regextasks #some list of tasks 