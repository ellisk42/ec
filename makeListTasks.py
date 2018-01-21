from type import guess_type, arrow
from task import RegressionTask
from utilities import eprint

from random import randint

import listroutines as lr


def hashable(v):
    """Determine whether `v` can be hashed."""
    try:
        hash(v)
    except TypeError:
        return False
    return True


def list_features(examples):
    # assume all tasks have the same number of examples
    # and all inputs are lists
    features = []
    ot = type(examples[0][1])
    mean = lambda l: 0 if not l else sum(l)/len(l)
    imean = [mean(i) for (i,), o in examples]
    ivar = [sum((v - imean[idx])**2
                for v in examples[idx][0][0])
            for idx in xrange(len(examples))]

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
        omean = [mean(o) for (i,), o in examples]
        ovar = [sum((v - omean[idx])**2
                    for v in examples[idx][1])
                for idx in xrange(len(examples))]
        cntr = lambda l, o: 0 if not l else len(set(l).difference(set(o))) / len(l)
        cnt_not_in_output = [cntr(i, o) for (i,), o in examples]

        #features += [len(i) for (i,), o in examples]
        #features += [len(o) for (i,), o in examples]
        features.append(sum(len(i) - len(o) for (i,), o in examples))
        #features += cnt_not_int_output
        features.append(sum(cnt_not_in_output))
        features.append(sum(om - im for im, om in zip(imean, omean)))
        features.append(sum(ov - iv for iv, ov in zip(ivar, ovar)))
        features.append(1)
        #features += [-1 for _ in examples]
        #features += [0 for _ in examples]
    elif ot == bool:
        outs = [o for (i,), o in examples]

        #features += [len(i) for (i,), o in examples]
        #features += [-1 for _ in examples]
        features.append(sum(len(i) for (i,), o in examples))
        #features += [0 for _ in examples]
        features.append(0)
        features.append(sum(imean))
        features.append(sum(ivar))
        features.append(-1)
        #features += [-1 for _ in examples]
        #features += outs
    else:  # int
        cntr = lambda l, o: 0 if not l else len(set(l).difference(set(o))) / len(l)
        cnt_not_in_output = [cntr(i, [o]) for (i,), o in examples]
        outs = [o for (i,), o in examples]

        #features += [len(i) for (i,), o in examples]
        #features += [1 for (i,), o in examples]
        features.append(sum(len(i) for (i,), o in examples))
        #features += cnt_not_int_output
        features.append(sum(cnt_not_in_output))
        features.append(sum(o - im for im, o in zip(imean, outs)))
        features.append(sum(ivar))
        features.append(0)
        #features += outs
        #features += [0 for _ in examples]

    return features


def make_list_task(routine, examples, **params):
    i, o = examples[0][0][0], examples[0][1]
    input_type = guess_type(i)
    output_type = guess_type(o)
    program_type = arrow(input_type, output_type)
    if type(i) == list:
        features = list_features(examples)
    else:
        features = []
    cache = hashable(i) and hashable(o)

    name = routine.id
    if params:
        eq_params = ["{}={}".format(k, v) for k, v in params.items()]
        if len(eq_params) == 1:
            ext = eq_params[0]
        elif len(eq_params) == 2:
            ext = "{} and {}".format(*eq_params)
        else:
            ext = ", ".join(eq_params[:-1])
            ext = "{}, and {}".format(ext, eq_params[-1])
        name += " with " + ext

    return RegressionTask(name, program_type, examples, features=features, cache=cache)


def make_list_tasks(n_examples=10):
    for routine in lr.find(count=100):  # all routines
        if routine.is_parametric():
            for params in routine.example_params():
                bigs = [k for k, v in params.items()
                        if type(v) == int and abs(v) > 9]
                for k in bigs:  # reduce big constants
                    params[k] = randint(1, 9)
                if routine.id == "rotate-k" and params["k"] != 0:
                    # rotate-k is hard if list is smaller than k
                    k = params["k"]
                    inps = []
                    for _ in xrange(n_examples):
                        r = randint(abs(k) + 1, 17)
                        inp = routine.gen(len=r, **params)[0]
                        inps.append(inp)
                else:
                    inps = routine.gen(count=n_examples, **params)
                examples = [((inp,), routine.eval(inp, **params))
                            for inp in inps]
                yield make_list_task(routine, examples, **params)
        else:
            inps = routine.examples()
            if len(inps) > n_examples:
                inps = inps[:n_examples]
            elif len(examples) < n_examples:
                inps += routine.gen(count=(n_examples - len(inps)))
            examples = [((inp,), routine.eval(inp)) for inp in inps]

            yield make_list_task(routine, examples)


def main():
    import cPickle as pickle

    eprint("Downloading and generating dataset")
    tasks = list(make_list_tasks())
    eprint("Got {} list tasks".format(len(tasks)))

    with open("data/list_tasks.pkl", "w") as f:
        pickle.dump(tasks, f)
    eprint("Wrote list tasks to data/list_tasks.pkl")


if __name__ == "__main__":
    main()
