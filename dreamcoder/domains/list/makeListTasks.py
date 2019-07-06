

from dreamcoder.type import *
from dreamcoder.task import Task
from dreamcoder.utilities import eprint, hashable

from random import randint, random, seed
from itertools import product

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

# These are tasks that are easy (solved from base DSL) and also uninteresting
# We exclude these from the test set
EASYLISTTASKS = {
    "add-k with k=2",
    "bool-identify-geq-k with k=2",
    "bool-identify-geq-k with k=3",
    "bool-identify-is-mod-k with k=1",
    "bool-identify-is-prime",
    "bool-identify-k with k=0",
    "bool-identify-k with k=1",
    "bool-identify-k with k=2",
    "caesar-cipher-k-modulo-n with k=3 and n=2",
    "drop-k with k=1",
    "drop-k with k=2",
    "drop-k with k=4",
    "index-head",
    "index-k with k=2",
    "index-k with k=4",
    "is-mod-k with k=1",
    "is-odds",
    "is-squares",
    "pow-k with k=2",
    "pow-k with k=3",
    "prepend-index-k with k=3",
    "prepend-index-k with k=5",
    "prepend-k with k=1",
    "prepend-k with k=2",
    "prepend-k with k=3",
    "remove-index-k with k=1",
    "replace-all-with-index-k with k=2",
    "replace-all-with-index-k with k=3",
    "slice-k-n with k=1 and n=2",
    "slice-k-n with k=2 and n=1",
    "slice-k-n with k=3 and n=1",
}

def make_list_task(name, examples, **params):
    input_type = guess_type([i for (i,), _ in examples])
    output_type = guess_type([o for _, o in examples])

    # We can internally handle lists of bools.
    # We explicitly create these by modifying existing routines.
    if name.startswith("identify"):
        boolexamples = [((i,), list(map(bool, o))) for (i,), o in examples]
        yield from make_list_task("bool-" + name, boolexamples, **params)
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
    import listroutines as lr

    for routine in lr.find(count=100):  # all routines
        if routine.id in EXCLUDES:
            continue
        if routine.is_parametric():
            keys = list(routine.example_params()[0].keys())
            for params in map(lambda values: dict(zip(keys, values)),
                              product(range(6), repeat=len(keys))):
                try:
                    if routine.id == "rotate-k":
                        # rotate-k is hard if list is smaller than k
                        k = params["k"]
                        if k < 1:
                            continue
                        inps = []
                        for _ in range(n_examples):
                            r = randint(abs(k) + 1, 17)
                            inp = routine.gen(len=r, **params)[0]
                            inps.append(inp)
                    else:
                        inps = routine.gen(count=n_examples, **params)
                    examples = [((inp,), routine.eval(inp, **params))
                                for inp in inps]
                    yield from make_list_task(routine.id, examples, **params)
                except lr.APIError:  # invalid params
                    continue
        else:
            inps = routine.examples()
            if len(inps) > n_examples:
                inps = inps[:n_examples]
            elif len(inps) < n_examples:
                inps += routine.gen(count=(n_examples - len(inps)))
            examples = [((inp,), routine.eval(inp)) for inp in inps]
            yield from make_list_task(routine.id, examples)


def make_list_bootstrap_tasks():
    seed(42)

    def suffixes(l):
        if l == []:
            return []
        else:
            return [l[1:]] + suffixes(l[1:])

    def flip(): return random() > 0.5

    def randomSuffix():
        return [randint(0, 9) for _ in range(randint(1, 4))]

    def randomList(minimum=0, minimumLength=4, maximumLength=6):
        return [randint(minimum, 9) for _ in range(randint(minimumLength, maximumLength))]

    def randomListOfLists():
        return [randomSuffix() for _ in range(randint(2, 4))]

    def randomListOfLists_bool(l=None):
        if l is None:
            l = randint(4, 7)
        return [randomBooleanList() for _ in range(l)]

    def randomBooleanList():
        return [flip() for _ in range(randint(4, 7))]

    # Reliably learned in under a minute; always triggers learning of length
    # primitive
    lengthBootstrap = [
        # Task("length bool", arrow(tlist(tbool), tint),
        #      [((l,), len(l))
        #       for _ in range(10)
        #       for l in [[flip() for _ in range(randint(0, 10))]]]),
        Task("length int", arrow(tlist(tint), tint),
             [((l,), len(l))
              for _ in range(10)
              for l in [randomList()]]),
        Task("map length", arrow(tlist(tlist(tint)), tlist(tint)),
             [((xss,), [len(xs) for xs in xss])
              for _ in range(10)
              for xss in [randomListOfLists()] ])
    ]

    # Encourages learning of unfolding
    unfoldBootstrap = [
        Task("countdown", arrow(tint, tlist(tint)),
             [((n,), list(range(n + 1, 1, -1)))
              for n in range(10)]),
        Task("weird count", arrow(tint, tlist(tint)),
             [((n,), list(range(-n,0,-1)))
              for n in range(-10,0) ]),
        Task("take every other", arrow(tlist(tint),tlist(tint)),
             [((l,), [x for j,x in enumerate(l) if j%2 == 0])
              for _ in range(9)
              for l in [ [randint(0, 9) for _ in range(randint(1,4)*2)] ] ] + [(([],),[])]),
        # Task("stutter every other", arrow(tlist(tint),tlist(tint)),
        #      [((l,), [l[int(j/2)] for j in range(len(l)) ])
        #       for _ in range(10)
        #       for l in [ [randint(0, 9) for _ in range(randint(1,4)*2)] ] ]),
        # Task("take until 3 reached", arrow(tlist(tint),tlist(tint)),
        #      [((p + [3] + s,),p)
        #       for _ in range(10)
        #       for p in [ [z for z in randomList()[:5] if z != 3 ]]
        #       for s in [randomList()] ]),
        Task("drop last element", arrow(tlist(tint),tlist(tint)),
             [((l,), l[:-1])
              for _ in range(10)
              for l in [ [randint(0, 9) for _ in range(randint(2,5))] ] ]),
        # Task("suffixes", arrow(tlist(tint), tlist(tlist(tint))),
        #      [((l,), suffixes(l))
        #       for _ in range(10)
        #       for l in [randomList()]]),
        Task("range", arrow(tint, tlist(tint)),
             [((n,), list(range(n)))
              for n in range(10)]),
        Task("range inclusive", arrow(tint, tlist(tint)),
             [((n,), list(range(n + 1)))
              for n in range(10)]),
        # Task("range inclusive+1", arrow(tint, tlist(tint)),
        #      [((n,), list(range(n + 2)))
        #       for n in range(10)]),
        # Task("range exclusive", arrow(tint, tlist(tint)),
        #      [((n,), list(range(n - 1)))
        #       for n in range(2, 11)]),
        # Task("range length", arrow(tlist(tint),tlist(tint)),
        #      [((l,),list(range(len(l))))
        #       for _ in range(10)
        #       for l in [randomList()] ])
    ]

    # Encourages learning how to treat a list as an array
    arrayBootstrap = [
        Task("index int", arrow(tint, tlist(tint), tint),
             [((n, l), l[n])
              for n in range(10)
              for l in [[randint(0, 9) for _ in range(randint(n + 1, n + 5))]]]),
        # Task("last n", arrow(tint, tlist(tint), tlist(tint)),
        #      [((n, l), l[-n:])
        #       for n in range(10)
        #       for l in [[randint(0, 9) for _ in range(randint(n + 1, n + 5))]]]),
        Task("1-index int", arrow(tint, tlist(tint), tint),
             [((n, l), l[n - 1])
              for n in range(1,11)
              for l in [[randint(0, 9) for _ in range(randint(n + 1, n + 4))]]])
        
        # Task("index bool", arrow(tint, tlist(tbool), tbool),
        #      [((n, l), l[n])
        #       for n in range(10)
        #       for l in [[flip() for _ in range(randint(n + 1, n + 5))]]])
    ]

    # Teaches how to slice lists, not sure if we really need this though
    sliceBootstrap = [
        Task("take bool", arrow(tint, tlist(tbool), tlist(tbool)),
             [((n, l), l[:n])
              for n in range(10)
              for l in [[flip() for _ in range(randint(n, n + 5))]]]),
        Task("drop bool", arrow(tint, tlist(tbool), tlist(tbool)),
             [((n, l), l[n:])
              for n in range(10)
              for l in [[flip() for _ in range(randint(n, n + 5))]]]),

        Task("take int", arrow(tint, tlist(tint), tlist(tint)),
             [((n, l), l[:n])
              for n in range(10)
              for l in [[randint(0, 9) for _ in range(randint(n, n + 5))]]]),
        Task("drop int", arrow(tint, tlist(tint), tlist(tint)),
             [((n, l), l[n:])
              for n in range(10)
              for l in [[randint(0, 9) for _ in range(randint(n, n + 5))]]]),

    ]

    # learning to fold
    foldBootstrap = [
        Task("stutter", arrow(tlist(tint),tlist(tint)),
             [((l,), [z for x in l for z in [x,x] ])
              for _ in range(10)
              for l in [randomList()] ]),
        Task("sum", arrow(tlist(tint), tint),
             [((l,), sum(l))
              for _ in range(10)
              for l in [randomList()]]),
        # Task("difference", arrow(tlist(tint), tint),
        #      [((l,), reduce(lambda x, y: y - x, reversed(l), 1))
        #       for _ in range(10)
        #       for l in [randomList()[:4]]]),
        # Task("append bool", arrow(tlist(tbool), tlist(tbool), tlist(tbool)),
        #      [((x, y), x + y)
        #       for _ in range(10)
        #       for [x, y] in [[randomBooleanList(), randomBooleanList()]]]),
        Task("append constant 0", arrow(tlist(tint),tlist(tint)),
             [((l,),l + [0])
              for _ in range(10)
              for l in [randomList()] ]),
    ]

    # learning to map
    mapBootstrap = [
        Task("map double", arrow(tlist(tint), tlist(tint)),
             [((l,), list(map(lambda n: n * 2, l)))
              for _ in range(10)
              for l in [randomList()]]),
        Task("map increment", arrow(tlist(tint),tlist(tint)),
             [((l,),list(map(lambda n: n+1, l)))
              for _ in range(10)
              for l in [randomList()] ]),
        Task("map negation", arrow(tlist(tint),tlist(tint)),
             [((l,),list(map(lambda n: 0-n, l)))
              for _ in range(10)
              for l in [randomList()] ]),
        # Task("map car", arrow(tlist(tlist(tint)), tlist(tint)),
        #      [((l,), [n[0] for n in l])
        #       for _ in4 range(10)
        #       for l in [randomListOfLists()]]),
        # Task("map cdr", arrow(tlist(tlist(tbool)),tlist(tlist(tbool))),
        #      [((l,),map(lambda n: n[1:],l))
        #       for _ in range(10)
        #       for l in [randomListOfLists_bool()]]),
        # Task("map empty?", arrow(tlist(tlist(tint)), tlist(tboolean)),
        #      [((l,), [n == [] for n in l])
        #       for _ in range(10)
        #       for l in [[[] if flip() else randomList() for _ in range(randint(1, 5))]]]),

        # Task("map eq 0?", arrow(tlist(tint),tlist(tboolean)),
        #      [((l,),map(lambda n: 0 == n,l))
        #       for _ in range(10)
        #       for l in [[ randint(0,3) for _ in range(randint(4,7)) ]] ])

    ]
    difficultMaps = [
                Task("map quadruple", arrow(tlist(tint), tlist(tint)),
             [((l,), list(map(lambda n: n * 4, l)))
              for _ in range(10)
              for l in [randomList()]]),
        Task("map add 3", arrow(tlist(tint),tlist(tint)),
             [((l,),list(map(lambda n: n+3, l)))
              for _ in range(10)
              for l in [randomList()] ]),

        ]

    # Learning to zip lists together
    zipBootstrap = [
        Task("zip plus", arrow(tlist(tint),tlist(tint),tlist(tint)),
             [((l1,l2),list(map(lambda x,y: x+y,l1,l2)))
              for _ in range(10)
              for l1 in [randomList(minimumLength=2, maximumLength=4)]
              for l2 in [[ randint(0,9) for _ in range(len(l1)) ]]]),
        Task("zip minus", arrow(tlist(tint),tlist(tint),tlist(tint)),
             [((l1,l2),list(map(lambda x,y: x-y,l1,l2)))
              for _ in range(10)
              for l1 in [randomList(minimumLength=2, maximumLength=4)]
              for l2 in [[ randint(0,9) for _ in range(len(l1)) ]]]),
        # Task("zip eq?", arrow(tlist(tint), tlist(tint), tlist(tbool)),
        #      [((l1, l2), list(map(lambda x, y: x == y, l1, l2)))
        #       for _ in range(10)
        #       for l1 in [[randint(0, 3) for _ in range(randint(4, 7))]]
        #       for l2 in [[randint(0, 3) for _ in range(len(l1))]]]),
        # Task("zip cons", arrow(tlist(tbool), tlist(tlist(tbool)), tlist(tlist(tbool))),
        #      [((l1, l2), list(map(lambda x, y: [x] + y, l1, l2)))
        #       for _ in range(10)
        #       for l1 in [randomBooleanList()]
        #       for l2 in [randomListOfLists_bool(l=len(l1))]]),
        # Task("zip cons", arrow(tlist(tint),tlist(tlist(tint)),tlist(tlist(tint))),
        #      [((l1,l2),list(map(lambda x,y: [x]+y,l1,l2)))
        #       for _ in range(10)
        #       for l1 in [randomList()]
        #       for l2 in [[ randomList() for _ in range(len(l1)) ]]]),
    ]

    # Learning to filter
    filterBootstrap = [
        # Task("remove empty lists",
        #      arrow(tlist(tlist(tbool)), tlist(tlist(tbool))),
        #      [((ls,), [l for l in ls if len(l) > 0])
        #       for _ in range(10)
        #       for ls in [[[flip() for _ in range(randint(0, 3))]
        #                   for _ in range(4)]]])
        # Task("remove non 0s",
        #      arrow(tlist(tint), tlist(tint)),
        #      [((xs,), filter(lambda x: x == 0, xs))
        #       for _ in range(10)
        #       for xs in [[ randint(0,3) for _ in range(5) ]] ]),
        Task("remove 0s",
             arrow(tlist(tint), tlist(tint)),
             [((xs,), [x for x in xs if x != 0])
              for _ in range(10)
              for xs in [[randint(0, 3) for _ in range(5)]]]),
        Task("remove non-positives",
             arrow(tlist(tint), tlist(tint)),
             [((xs,), [x for x in xs if not (x > 1)])
              for _ in range(10)
              for xs in [[randint(0, 3) for _ in range(5)]]]),
    ]

    return lengthBootstrap + filterBootstrap + \
        unfoldBootstrap + arrayBootstrap + foldBootstrap + mapBootstrap + zipBootstrap


def bonusListProblems():
    # Taken from https://www.ijcai.org/Proceedings/75/Papers/037.pdf
    # These problems might be a lot easier if we do not use numbers
    def randomList(lb=None, ub=None):
        if lb is None:
            lb = 2
        if ub is None:
            ub = 5
        return [randint(0, 5) for _ in range(randint(lb, ub))]

    bonus = [
        Task(
            "pair reverse", arrow(tlist(tint), tlist(tint)),
            [((x,), [x[j + (1 if j % 2 == 0 else -1)]
                     for j in range(len(x))])
             for _ in range(5)
             for x in [randomList(10, 10)]]
        ),
        Task(
            "duplicate each element", arrow(tlist(tint), tlist(tint)),
            [((x,), [a for z in x for a in [z] * 2])
             for _ in range(5)
             for x in [randomList(4, 6)]]
        ),
        Task(
            "reverse duplicate each element", arrow(tlist(tint), tlist(tint)),
            [((x,), [a for z in reversed(x) for a in [z] * 2])]
        ),
    ]
    return bonus

def sortBootstrap():
    # These tasks have as their goal the learning of (1) filter, and
    # (2) sort, which uses filter.
    def flip(): return random() > 0.5
    def randomList(lb=None, ub=None):
        if lb is None:
            lb = 2
        if ub is None:
            ub = 5
        return [randint(0, 10) for _ in range(randint(lb, ub))]
    def randomBooleanList():
        return [flip() for _ in range(randint(4, 7))]
    def removeDuplicates(l):
        if len(l) == 0: return l
        return [l[0]] + removeDuplicates([ z for z in l if z != l[0] ])
    
    filterBootstrap = [
        # Task("remove empty lists",
        #      arrow(tlist(tlist(tbool)), tlist(tlist(tbool))),
        #      [((ls,), [l for l in ls if len(l) > 0])
        #       for _ in range(10)
        #       for ls in [[[flip() for _ in range(randint(0, 3))]
        #                   for _ in range(4)]]]),
        # Task("remove non 0s",
        #      arrow(tlist(tint), tlist(tint)),
        #      [((xs,), filter(lambda x: x == 0, xs))
        #       for _ in range(10)
        #       for xs in [[ randint(0,3) for _ in range(5) ]] ]),
        Task("remove 0s",
             arrow(tlist(tint), tlist(tint)),
             [((xs,), [x for x in xs if x != 0])
              for _ in range(10)
              for xs in [[randint(0, 3) for _ in range(5)]]]),
        # Task("remove primes",
        #      arrow(tlist(tint), tlist(tint)),
        #      [((xs,), [x for x in xs if not (x in {2,3,5,7,11,13,17,19,23})])
        #       for _ in range(10)
        #       for xs in [[randint(0, 20) for _ in range(7)]]]),
        Task("remove squares",
             arrow(tlist(tint), tlist(tint)),
             [((xs,), [x for x in xs if not (int(x**0.5)**2 == x)])
              for _ in range(10)
              for xs in [[randint(0, 20) for _ in range(7)]]]),
        Task("remove > 1",
             arrow(tlist(tint), tlist(tint)),
             [((xs,), [x for x in xs if not (x > 1)])
              for _ in range(10)
              for xs in [[randint(0, 5) for _ in range(7)]]]),
    ]

    # Needed for selection sort
    minimumBootstrap = [
        Task("min2", arrow(tint,tint,tint),
             [((x,y),min(x,y))
              for x in range(4)
              for y in range(4) ]),
        Task("minimum of list", arrow(tlist(tint),tint),
             [((l,),min(l))
              for _ in range(15) 
              for l in [randomList()] ])
    ]

    appendBootstrap = [
        Task("append bool", arrow(tlist(tbool), tlist(tbool), tlist(tbool)),
             [((x, y), x + y)
              for _ in range(10)
              for [x, y] in [[randomBooleanList(), randomBooleanList()]]]),
        Task("append int", arrow(tlist(tint), tlist(tint), tlist(tint)),
             [((x, y), x + y)
              for _ in range(10)
              for [x, y] in [[randomList(), randomList()]]])
    ]

    insertionBootstrap = [
        Task("filter greater than or equal", arrow(tint,tlist(tint),tlist(tint)),
             [((x,l), [y for y in l if y >= x ])
              for _ in range(15) 
              for x in [randint(0,5)]
              for l in [randomList()] ]),
        Task("filter less than", arrow(tint,tlist(tint),tlist(tint)),
             [((x,l), [y for y in l if y < x ])
              for _ in range(15) 
              for x in [randint(0,5)]
              for l in [randomList()] ]),
        Task("insert into sorted list (I)", arrow(tint,tlist(tint),tlist(tint)),
             [((x,l), [y for y in l if y < x ] + [x] + [y for y in l if y >= x ])
              for _ in range(15) 
              for x in [randint(0,5)]
              for _l in [randomList()]
              for l in [sorted(_l)] ]),
        Task("insert into sorted list (II)", arrow(tint,tlist(tint),tlist(tint)),
             [((x,l), [y for y in l if y < x ] + [x] + [y for y in l if y >= x ])
              for _ in range(15) 
              for x in [randint(0,5)]
              for l in [randomList()] ])
    ]


    sortTask = [
        Task("sort-and-deduplicate", arrow(tlist(tint),tlist(tint)),
             [((l,),list(sorted(l)))
              for _ in range(15)
              for l in [removeDuplicates(randomList())]
             ])]

    slowSort = [
        Task("+1 maximum list", arrow(tlist(tint), tint),
             [((l,),max(l) + 1)
              for _ in range(15)
              for l in [randomList()] ]),
        Task("range +1 maximum list", arrow(tlist(tint), tlist(tint)),
             [((l,),list(range(max(l) + 1)))
              for _ in range(15)
              for l in [randomList()] ]),
        ]
        

    tasks = sortTask + slowSort
    for t in tasks: t.mustTrain = True
    return tasks
    

def exportTasks():
    import sys
    import pickle as pickle

    n_examples = 15
    if len(sys.argv) > 1:
        n_examples = int(sys.argv[1])

    eprint("Downloading and generating dataset")
    tasks = sorted(make_list_tasks(n_examples), key=lambda t: t.name)
    eprint("Got {} list tasks".format(len(tasks)))

    with open("data/list_tasks.pkl", "w") as f:
        pickle.dump(tasks, f)
    eprint("Wrote list tasks to data/list_tasks.pkl")


if __name__ == "__main__":
    import json
    def retrieveJSONTasks(filename, features=False):
        """
        For JSON of the form:
            {"name": str,
             "type": {"input" : bool|int|list-of-bool|list-of-int,
                      "output": bool|int|list-of-bool|list-of-int},
             "examples": [{"i": data, "o": data}]}
        """
        with open(filename, "r") as f:
            loaded = json.load(f)
        TP = {
            "bool": tbool,
            "int": tint,
            "list-of-bool": tlist(tbool),
            "list-of-int": tlist(tint),
        }
        return [Task(
            item["name"],
            arrow(TP[item["type"]["input"]], TP[item["type"]["output"]]),
            [((ex["i"],), ex["o"]) for ex in item["examples"]],
            features=(None if not features else list_features(
                [((ex["i"],), ex["o"]) for ex in item["examples"]])),
            cache=False,
        ) for item in loaded]
    for t in retrieveJSONTasks("data/list_tasks.json") + sortBootstrap() + make_list_bootstrap_tasks():
        print(t.describe())
        print()
    # exportTasks()
