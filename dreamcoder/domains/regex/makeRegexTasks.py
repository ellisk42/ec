import dill
import os
import json
from string import printable

from pregex import pregex

from dreamcoder.task import Task
from dreamcoder.type import tpregex, arrow
from dreamcoder.utilities import get_data_dir


def makeOldTasks():
    # a series of tasks

    taskfile = os.path.join(get_data_dir(), 'data_filtered.json')
    #task_list = pickle.load(open(taskfile, 'rb'))

    with open(taskfile) as f:
        file_contents = f.read()
    task_list = json.loads(file_contents)


    # if I were to just dump all of them:
    regextasks = [
        Task("Luke data column no." + str(i),
             arrow(tpregex, tpregex),
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

    taskfile = os.path.join(get_data_dir(), "regex_data_csv_900.p")

    with open(taskfile, 'rb') as handle:
        data = dill.load(handle)

    tasklist = data[0][:100] #a list of indices

    regextasks = [
        Task("Data column no. " + str(i),
            arrow(tpregex, tpregex),
            [((), example) for example in task] 
        ) for i, task in enumerate(tasklist)]



    return regextasks

def makeLongTasks():

    #load new data:

    taskfile = os.path.join(get_data_dir(), "regex_data_csv_900.p")

    with open(taskfile, 'rb') as handle:
        data = dill.load(handle)

    tasklist = data[0] #a list of indices

    regextasks = [
        Task("Data column no. " + str(i),
            arrow(tpregex, tpregex),
            [((), example) for example in task] 
        ) for i, task in enumerate(tasklist)]



    return regextasks

def makeWordTasks():

    #load new data:

    taskfile = os.path.join(get_data_dir(), "regex_data_csv_900.p")

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
            arrow(tpregex, tpregex),
            [((), example) for example in task] 
        ) for i, task in enumerate(tasklist) if i in full_list ]

    for i in train_list:
        regextasks[i].mustTrain = True


    return regextasks

def makeNumberTasks():

    #load new data:

    taskfile = os.path.join(get_data_dir(), "regex_data_csv_900.p")

    with open(taskfile, 'rb') as handle:
        data = dill.load(handle)

    tasklist = data[0] #a list of indices



    #match_col(data[0],'\d*\.\d*')
    raw_decimals = [121, 122, 163, 164, 165, 170, 172, 173, 175, 178, 218, 228, 230, 231, 252, 253,
    254, 258, 259, 305, 320, 330, 334, 340, 348, 350, 351, 352, 353, 355, 357, 358, 361, 363, 364, 
    371, 380, 382, 409, 410, 411, 447, 448, 449, 450, 458, 469, 471, 533, 562, 564]


    decimals_pos_neg_dollar = [3, 4, 5, 6, 7, 13, 16, 24, 27, 28, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 
    53, 54, 55, 57, 58, 60, 61, 63, 64, 65, 66, 68, 69, 70, 71, 73, 74, 77, 78, 80, 81, 103, 104, 105, 
    106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 125, 126, 128,
     129, 131, 132, 134, 135, 139, 146, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,
      166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 180, 181, 182, 183, 184, 185, 186,
       193, 194, 195, 204, 205, 207, 209, 210, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 223, 224,
        225, 226, 227, 228, 229, 230, 231, 232, 249, 250, 251, 252, 253, 254, 255, 256, 258, 259, 260, 261,
         263, 266, 267, 270, 271, 272, 277, 299, 301, 302, 305, 306, 307, 309, 312, 313, 315, 319, 320, 324,
          326, 327, 330, 334, 340, 348, 350, 351, 352, 353, 354, 355, 356, 357, 358, 361, 362, 363, 364, 368,
           371, 373, 377, 380, 382, 400, 401, 402, 403, 405, 406, 409, 410, 411, 413, 435, 439, 446, 447, 448,
            449, 450, 451, 452, 453, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 469, 470, 471, 477,
             498, 500, 502, 503, 507, 512, 518, 519, 520, 532, 533, 553, 554, 555, 556, 557, 558, 559, 560, 561,
              562, 564, 565, 572, 577]

    #match_col(data[0],'(\d*,?\d*)+')
    commas = []
    #match_col(data[0],'(\d*,?\d*)+')
    commas_and_all = []

    #full_list = test_list + train_list
    train_list = []
    full_list = decimals_pos_neg_dollar

    regextasks = [
        Task("Data column no. " + str(i),
            arrow(tpregex, tpregex),
            [((), example) for example in task] 
        ) for i, task in enumerate(tasklist) if i in full_list ]

    for i in train_list:
        regextasks[i].mustTrain = True


    return regextasks


def makeHandPickedTasks():

    #load new data:

    taskfile = os.path.join(get_data_dir(), "regex_data_csv_900.p")

    with open(taskfile, 'rb') as handle:
        data = dill.load(handle)

    tasklist = data[0] #a list of indices


    full_list = list(range(199)) + \
    [209,218,222,223,224,225,226] + \
    list(range(222,233)) + \
    [235,237,238,239,243,244,245,252,253,254,255,257,258,259,260,261,264,265,269,272,274] + \
    list(range(275,291)) + \
    [295,297,300,303,304,305,306,310,311,312,314,315,316,320,321,323,327,329,330,333,334,335,337,338,339,340,341,342,343,344] + \
    list(range(348,359)) + \
    [361,369,373,379,380,382,387,403,405,407,408] + \
    list(range(409,417)) + \
    list(range(418,437)) + \
    list(range(440,444)) + \
    list(range(446,452)) + \
    list(range(456,460)) + \
    list(range(466,472)) + \
    [503,504]


    regextasks = [
        Task("Data column no. " + str(i),
            arrow(tpregex, tpregex),
            [((), example) for example in task] 
        ) for i, task in enumerate(tasklist) if i in full_list ]

    #for i in train_list:
    #    regextasks[i].mustTrain = True


    return regextasks

def makeNewTasks(include_only=None):

    #load new data:

    taskfile = os.path.join(get_data_dir(), "csv_filtered_all_background_novel.p")

    with open(taskfile, 'rb') as handle:
        data = dill.load(handle)

    tasklist = data['background'] #a list of indices

    if include_only:
        regextasks = [
            Task("Data column no. " + str(i),
                arrow(tpregex, tpregex),
                [((), example) for example in task['train']] 
            ) for i, task in enumerate(tasklist) if i in include_only]
    else:
        regextasks = [
            Task("Data column no. " + str(i),
                arrow(tpregex, tpregex),
                [((), example) for example in task['train']] 
            ) for i, task in enumerate(tasklist)]

    #for i in train_list:
    #    regextasks[i].mustTrain = True

    return regextasks
REGEXTASKS = None
def regexHeldOutExamples(task, include_only=None):

    #load new data:
    global REGEXTASKS
    if REGEXTASKS is None:
        taskfile = os.path.join(get_data_dir(), "csv_filtered_all_background_novel.p")

        with open(taskfile, 'rb') as handle:
            data = dill.load(handle)

        tasklist = data['background'] #a list of indices

        if include_only:
            regextasks = [
                Task("Data column no. " + str(i),
                    arrow(tpregex, tpregex),
                    [((), example) for example in _task['test']] 
                ) for i, _task in enumerate(tasklist) if i in include_only]
        else:
            regextasks = [
                Task("Data column no. " + str(i),
                    arrow(tpregex, tpregex),
                    [((), example) for example in _task['test']] 
                ) for i, _task in enumerate(tasklist)]

        #for i in train_list:
        #    regextasks[i].mustTrain = True

        REGEXTASKS = {t.name: t.examples for t in regextasks}
    fullTask = REGEXTASKS[task.name]
    return fullTask
        


def makeNewNumberTasks():

    tasks = makeNewTasks()
    numberTasks = [t for t in tasks if not any(p in ex for p in printable[10:62] for _, ex in t.examples)]
    return numberTasks 


# a helper function which takes a list of lists and sees which match a specific regex.
def match_col(dataset, rstring):
    r = pregex.create(rstring)
    matches = []
    for i, col in enumerate(dataset):
        score = sum([r.match(example) for example in col])
        if score != float('-inf'):
            matches.append(i)
    return matches

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_only",
                        default=None,
                        nargs="+",
                        type=int)
    args = parser.parse_args()


    def show_tasks(dataset):
        task_list = []
        for task in dataset:
            print(task.name)
            print([example[1] for example in task.examples[:20]])
            task_list.append([example[1] for example in task.examples])
        return task_list

    task = {"number": makeNumberTasks,
    "words": makeWordTasks,
    "all": makeLongTasks,
    "new": makeNewTasks}['new']


    x = show_tasks(task(args.include_only))

    
