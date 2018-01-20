from ec import explorationCompression, commandlineArguments
from utilities import eprint
from listPrimitives import primitives
import cPickle as pickle

def apply_list_features(task):
    # assume all tasks have the same number of examples
    # and all inputs are lists
    features = []
    ot = type(task.examples[0][1])
    mean = lambda l: 0 if not l else sum(l)/len(l)
    imean = [mean(i) for (i,), o in task.examples]
    ivar = [sum((v - imean[idx])**2
                for v in task.examples[idx][0][0])
            for idx in xrange(len(task.examples))]

    #DISABLED length of each input and output
    # total difference between length of input and output
    #DISABLED normalized count of numbers in input but not in output
    # total normalized count of numbers in input but not in output
    # total difference between means of input and output
    # total difference between variances of input and output
    # output type (-1=bool, 0=int, 1=list)
    #DISABLED outputs if integers, else -1s
    #DISABLED outputs if bools (-1/1), else 0s
    if ot == list:
        omean = [mean(o) for (i,), o in task.examples]
        ovar = [sum((v - omean[idx])**2
                    for v in task.examples[idx][1])
                for idx in xrange(len(task.examples))]
        cntr = lambda l, o: 0 if not l else len(set(l).difference(set(o))) / len(l)
        cnt_not_in_output = [cntr(i, o) for (i,), o in task.examples]

        #features += [len(i) for (i,), o in task.examples]
        #features += [len(o) for (i,), o in task.examples]
        features.append(sum(len(i) - len(o) for (i,), o in task.examples))
        #features += cnt_not_int_output
        features.append(sum(cnt_not_in_output))
        features.append(sum(om - im for im, om in zip(imean, omean)))
        features.append(sum(ov - iv for iv, ov in zip(ivar, ovar)))
        features.append(1)
        #features += [-1 for _ in task.examples]
        #features += [0 for _ in task.examples]
    elif ot == bool:
        outs = [o for (i,), o in task.examples]

        #features += [len(i) for (i,), o in task.examples]
        #features += [-1 for _ in task.examples]
        features.append(sum(len(i) for (i,), o in task.examples))
        #features += [0 for _ in task.examples]
        features.append(0)
        features.append(sum(imean))
        features.append(sum(ivar))
        features.append(-1)
        #features += [-1 for _ in task.examples]
        #features += outs
    else:  # int
        cntr = lambda l, o: 0 if not l else len(set(l).difference(set(o))) / len(l)
        cnt_not_in_output = [cntr(i, [o]) for (i,), o in task.examples]
        outs = [o for (i,), o in task.examples]

        #features += [len(i) for (i,), o in task.examples]
        #features += [1 for (i,), o in task.examples]
        features.append(sum(len(i) for (i,), o in task.examples))
        #features += cnt_not_int_output
        features.append(sum(cnt_not_in_output))
        features.append(sum(o - im for im, o in zip(imean, outs)))
        features.append(sum(ivar))
        features.append(0)
        #features += outs
        #features += [0 for _ in task.examples]

    task.features = features
    return task

if __name__ == "__main__":
    try:
        with open("data/list_tasks.pkl") as f:
            tasks = pickle.load(f)
    except Exception as e:
        from makeListTasks import main
        main()
        with open("data/list_tasks.pkl") as f:
            tasks = pickle.load(f)

    tasks = map(apply_list_features, tasks)

    eprint("Got {} list tasks".format(len(tasks)))

    explorationCompression(primitives, tasks,
                           outputPrefix="experimentOutputs/list",
                           **commandlineArguments(frontierSize=10**4,
                                                  a=1,
                                                  iterations=10,
                                                  pseudoCounts=10.0))

