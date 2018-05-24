from type import tpregex
from task import Task
from pregex import pregex
import pickle
import json
import dill





def makeOldTasks():
    # a series of tasks

    taskfile = './data_filtered.json'
    #task_list = pickle.load(open(taskfile, 'rb'))


    with open('./data_filtered.json') as f:
        file_contents = f.read()
    task_list = json.loads(file_contents)


    # if I were to just dump all of them:
    regextasks = [
        Task("Luke data column no." + str(i),
             tpregex,
                 [((), example) for example in task_list[i]]
             ) for i in range(len(task_list))]
    """ regextasks = [
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
    return regextasks  # some list of tasks


def makeShortTasks():

    #load new data:

    taskfile = "./regex_data_csv_900.p"

    with open(taskfile, 'rb') as handle:
        data = dill.load(handle)

    tasklist = data[0][:100] #a list of indices

    regextasks = [
        Task("Data column no. " + str(i),
            tpregex,
            [((), example) for example in task] 
        ) for i, task in enumerate(tasklist)]



    return regextasks

def makeLongTasks():

    #load new data:

    taskfile = "./regex_data_csv_900.p"

    with open(taskfile, 'rb') as handle:
        data = dill.load(handle)

    tasklist = data[0] #a list of indices

    regextasks = [
        Task("Data column no. " + str(i),
            tpregex,
            [((), example) for example in task] 
        ) for i, task in enumerate(tasklist)]



    return regextasks

def makeWordTasks():

    #load new data:

    taskfile = "./regex_data_csv_900.p"

    with open(taskfile, 'rb') as handle:
        data = dill.load(handle)

    tasklist = data[0] #a list of indices




    all_upper = [0, 2, 8, 9, 10, 11, 12, 17, 18, 19, 20, 22]
    all_lower = [1]

    # match_col(data[0],'\\u(\l+)')
    one_capital_lower_plus = [144, 200, 241, 242, 247, 296, 390, 392, 444, 445, 481, 483, 485, 489, 493, 542, 549, 550, 581]

    #match_col(data[0],'(\l ?)+')
    lower_with_maybe_spaces = [1, 42, 47, 99, 100, 102, 201, 246, 248, 293, 294, 345, 437, 545, 590]

    #match_col(data[0],'(\\u\l+ ?)+')
    capital_then_lower_maybe_spaces = [144, 200, 241, 242, 247, 296, 390, 392, 395, 438, 444, 445, 481, 483, 484, 485, 487, 489, 493, 494, 542, 546, 549, 550, 578, 581, 582, 588, 591, 624, 629]

    #match_col(data[0],'(\\u+ ?)+')
    all_caps_spaces = [0, 2, 8, 9, 10, 11, 12, 17, 18, 19, 20, 22, 25, 26, 35, 36, 43, 45, 46, 49, 50, 52, 56, 59, 87, 89, 95, 101, 140, 147, 148, 149, 199, 332, 336, 397, 491, 492, 495, 580, 610]

    #one_capital_and_lower = [566, 550, 549, 542, 505, 493, 494, 489, 488, 485, 483, 481, 445, 444, 438, 296, 241, 242, 200, ]
    #all_lower_with_a_space = [545]
    #all_lower_maybe_space = [534]
    #one_capital_lower_maybe_spaces = [259, 262, 263, 264]


    #full_list = test_list + train_list
    train_list = []
    full_list = all_upper + all_lower + one_capital_lower_plus + lower_with_maybe_spaces + capital_then_lower_maybe_spaces + all_caps_spaces

    regextasks = [
        Task("Data column no. " + str(i),
            tpregex,
            [((), example) for example in task] 
        ) for i, task in enumerate(tasklist) if i in full_list ]

    for i in train_list:
        regextasks[i].mustTrain = True


    return regextasks

def match_col(dataset, rstring):
    r = pregex.create(rstring)
    matches = []
    for i, col in enumerate(dataset):
        score = sum([r.match(example) for example in col])
        if score != float('-inf'):
            matches.append(i)
    return matches


    
