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


def make_list_task(routine, examples, **params):
    i, o = examples[0][0][0], examples[0][1]
    input_type = guess_type(i)
    output_type = guess_type(o)
    program_type = arrow(input_type, output_type)
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

    return RegressionTask(name, program_type, examples, cache=cache)


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
                        r = randint(0, abs(k) - 1)
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
