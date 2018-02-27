from __future__ import division

from type import *
from task import Task
from utilities import eprint, hashable

from random import randint
from itertools import product, izip, imap

import listroutines as lr

# Excluded routines either impossible or astronomically improbable
# I'm cutting these off at ~20 nats in learned grammars.
EXCLUDES = {
    "dedup",
    "intersperse-k",
    "pow-base-k",
    "prime",
    "replace-all-k-with-n",
    "replace-index-k-with-n",
    "uniq",
}


def make_list_task(name, examples, **params):
    input_type = guess_type([i for (i,), _ in examples])
    output_type = guess_type([o for _, o in examples])

    # We can internally handle lists of bools.
    # We explicitly create these by modifying existing routines.
    if name.startswith("identify"):
        boolexamples = [((i,), map(bool, o)) for (i,), o in examples]
        for t in make_list_task("bool-"+name, boolexamples, **params):
            yield t
        # for now, we'll stick with the boolean-only tasks and not have a copy
        # for integers.
        return

    program_type = arrow(input_type, output_type)
    cache = all(hashable(x) for x in examples)

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

    yield Task(name, program_type, examples, cache=cache)


def make_list_tasks(n_examples):
    for routine in lr.find(count=100):  # all routines
        if routine.id in EXCLUDES:
            continue
        if routine.is_parametric():
            keys = routine.example_params()[0].keys()
            for params in imap(lambda values: dict(izip(keys, values)),
                               product(xrange(6), repeat=len(keys))):
                try:
                    if routine.id == "rotate-k":
                        # rotate-k is hard if list is smaller than k
                        k = params["k"]
                        if k < 1:
                            continue
                        inps = []
                        for _ in xrange(n_examples):
                            r = randint(abs(k) + 1, 17)
                            inp = routine.gen(len=r, **params)[0]
                            inps.append(inp)
                    else:
                        inps = routine.gen(count=n_examples, **params)
                    examples = [((inp,), routine.eval(inp, **params))
                                for inp in inps]
                    for t in make_list_task(routine.id, examples, **params):
                        yield t
                except lr.APIError:  # invalid params
                    continue
        else:
            inps = routine.examples()
            if len(inps) > n_examples:
                inps = inps[:n_examples]
            elif len(inps) < n_examples:
                inps += routine.gen(count=(n_examples - len(inps)))
            examples = [((inp,), routine.eval(inp)) for inp in inps]
            for t in make_list_task(routine.id, examples):
                yield t


def make_list_bootstrap_tasks(numberOfExamples):
    def randomSuffix():
        return [ randint(0,9) for _ in range(randint(1,4)) ]
    filterBootstrap = []

    for name, f in [("square", lambda x: (int(x**0.5)**2 == x)),
                    ("prime", lambda x: x in {2,3,5,7}),
                    ("is even", lambda x: x%2 == 0),
                    ("is odd", lambda x: x%2 == 1),
                    ("is < 5", lambda x: x < 5),
                    ("is < 3", lambda x: x < 3),
                    ("is < 4", lambda x: x < 4),
                    ("is > 5", lambda x: x > 5),
                    ("is > 3", lambda x: x > 3),
                    ("is > 4", lambda x: x > 4),
                    ("is 2", lambda x: x == 2),
                    ("is 5", lambda x: x == 5),
                    ("is 3", lambda x: x == 3),]:
        t = Task("Prepend if %s"%name,
                 arrow(tint,tlist(tint),tlist(tint)),
                 [ ((x,s), [x]+s if f(x) else s)
                   for x in range(10)
                   for s in [randomSuffix()]
                 ])
        filterBootstrap.append(t)

    reverseBootstrap = []
    reverseBootstrap.append(Task("reverse",
                                 arrow(tlist(tint),tlist(tint)),
                                 [ ((x,), list(reversed(x)))
                                   for _ in range(10) 
                                   for x in [randomSuffix()] ]))
    reverseBootstrap.append(Task("sort backwards",
                                 arrow(tlist(tint),tlist(tint)),
                                 [ ((x,), list(reversed(sorted(x))))
                                   for _ in range(10) 
                                   for x in [randomSuffix()] ]))

    indexBootstrap = []
    from random import choice
    for j in range(6):
        t = Task("If i = %d then x else y"%j,
                 arrow(tint,tint,tint,tint),
                 [ ((i,x,a), x if i == j else a)
                    for i in range(6)
                    for x in [ randint(0,9) ]
                    for a in [ choice(list(set(range(10)) - {x})) ]
                 ])
        indexBootstrap.append(t)

    return filterBootstrap + reverseBootstrap + indexBootstrap
        
def main():
    import sys
    import cPickle as pickle

    n_examples = 15
    if len(sys.argv) > 1:
        n_examples = int(sys.argv[1])

    eprint("Downloading and generating dataset")
    tasks = sorted(make_list_tasks(n_examples), key=lambda t:t.name)
    eprint("Got {} list tasks".format(len(tasks)))

    with open("data/list_tasks.pkl", "w") as f:
        pickle.dump(tasks, f)
    eprint("Wrote list tasks to data/list_tasks.pkl")


if __name__ == "__main__":
    main()
