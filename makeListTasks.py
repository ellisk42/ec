from __future__ import division

from type import *
from task import Task
from utilities import eprint, hashable

from random import randint, random, seed
from itertools import product, izip, imap

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
    import listroutines as lr
    
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


def make_list_bootstrap_tasks():
    seed(42)

    def suffixes(l):
        if l == []:
            return []
        else:
            return [l[1:]] + suffixes(l[1:])

    def flip(): return random() > 0.5
    def randomSuffix():
        return [ randint(0,9) for _ in range(randint(1,4)) ]
    def randomList(minimum = 0):
        return [ randint(minimum,9) for _ in range(randint(4,7)) ]
    def randomListOfLists():
        return [ randomSuffix() for _ in range(randint(2,4)) ]
    def randomBooleanList():
        return [ flip() for _ in range(randint(4,7)) ]

    if False:
        return [        Task("all", arrow(tlist(tbool),tbool),
             [((l,), reduce(lambda x,y: x and y, l, True))
              for sz in range(10)
              for l in [[random() > 0.2 for _ in range(sz) ]] ]),
        Task("or", arrow(tboolean,tboolean,tboolean),
             [((a,b),a or b)
              for a in [True,False]
              for b in [True,False] ]),
        Task("or car", arrow(tlist(tboolean),tlist(tboolean),tboolean),
             [((a,b),a[0] or b[0])
              for _ in xrange(10) 
              for a in [randomBooleanList()]
              for b in [randomBooleanList()] ]),]

    # Reliably learned in under a minute; always triggers learning of length primitive
    lengthBootstrap = [
        Task("length bool", arrow(tlist(tbool),tint),
             [((l,), len(l))
              for _ in range(10)
              for l in [[flip() for _ in range(randint(0,10)) ]] ]),
        Task("length int", arrow(tlist(tint),tint),
             [((l,), len(l))
              for _ in range(10)
              for l in [randomList()] ]),
    ]

    # Encourages learning of simple nonrecursive operations that are useful in recursive functions
    operationBootstrap = [
        Task("increment", arrow(tint,tint),
             [((n,),n+1) for n in range(5) ]),
        Task("decrement", arrow(tint,tint),
             [((n,),n-1) for n in range(5) ]),
        Task("zero?", arrow(tint,tbool),
             [((n,), n == 0) for n in range(5) ]),
        Task("zero car?", arrow(tlist(tint),tbool),
             [(([h] + l,), h == 0)
              for _ in range(5)
              for h in [0,randint(1,9)]
              for l in [randomList()] ]),
        Task("increment car", arrow(tlist(tint),tint),
             [((l,),l[0] + 1)
              for _ in range(10)
              for l in [randomList()] ]),
        Task("increment twice", arrow(tint,tint),
             [((n,),n+2) for n in range(5) ]),
        Task("decrement twice", arrow(tint,tint),
             [((n,),n-2) for n in range(5) ]),
        Task("decrement car", arrow(tlist(tint),tint),
             [((l,),l[0] - 1)
              for _ in range(10)
              for l in [randomList()] ]),
    ]

    # Encourages learning of unfolding operations
    unfoldBootstrap = [
        Task("countdown", arrow(tint,tlist(tint)),
             [((n,), range(n+1,1,-1))
                              for n in range(10) ]),
        Task("suffixes", arrow(tlist(tint),tlist(tlist(tint))),
             [((l,), suffixes(l))
              for _ in xrange(10)
              for l in [randomList()] ]),
        Task("range", arrow(tint,tlist(tint)),
             [((n,), range(n))
              for n in range(10) ]),
        Task("range len", arrow(tlist(tboolean),tlist(tint)),
             [((l,), range(len(l)))
              for _ in range(10)
              for l in [randomBooleanList()] ]),
    ]
    
    # Encourages learning how to treat a list as an array
    arrayBootstrap = [
        Task("index int", arrow(tint, tlist(tint), tint),
             [((n,l), l[n])
              for n in range(10)
              for l in [[randint(0,9) for _ in range(randint(n+1,n + 5)) ]]]),
        Task("index bool", arrow(tint, tlist(tbool), tbool),
             [((n,l), l[n])
              for n in range(10)
              for l in [[flip() for _ in range(randint(n+1,n + 5)) ]]])
    ]
    
    # Teaches how to slice lists, not sure if we really need this though
    sliceBootstrap = [
        Task("take bool", arrow(tint, tlist(tbool), tlist(tbool)),
             [((n,l), l[:n])
              for n in range(10)
              for l in [[flip() for _ in range(randint(n,n + 5)) ]] ]),
        Task("drop bool", arrow(tint, tlist(tbool), tlist(tbool)),
             [((n,l), l[n:])
              for n in range(10)
              for l in [[flip() for _ in range(randint(n,n + 5)) ]] ]),

        Task("take int", arrow(tint, tlist(tint), tlist(tint)),
             [((n,l), l[:n])
              for n in range(10)
              for l in [[randint(0,9) for _ in range(randint(n,n + 5)) ]] ]),
        Task("drop int", arrow(tint, tlist(tint), tlist(tint)),
             [((n,l), l[n:])
              for n in range(10)
              for l in [[ randint(0,9) for _ in range(randint(n,n+5)) ]] ]),

    ]

    # learning to fold
    foldBootstrap = [        
        Task("sum", arrow(tlist(tint),tint),
             [((l,), sum(l))
              for _ in range(10)
              for l in [randomList()] ]),
        Task("difference", arrow(tlist(tint),tint),
             [((l,), reduce(lambda x,y: y-x, reversed(l), 1))
              for _ in range(10)
              for l in [randomList()[:4]] ]),
        Task("append bool", arrow(tlist(tbool),tlist(tbool),tlist(tbool)),
             [((x,y), x+y)
              for _ in range(10)
              for [x,y] in [[randomBooleanList(),randomBooleanList()]] ])
    ]

    # learning to map
    mapBootstrap = [
        Task("map double", arrow(tlist(tint),tlist(tint)),
             [((l,),map(lambda n: n*2, l))
              for _ in range(10)
              for l in [randomList()] ]),
        # Task("map increment", arrow(tlist(tint),tlist(tint)),
        #      [((l,),map(lambda n: n+1, l))
        #       for _ in range(10)
        #       for l in [randomList()] ]),
        Task("map negation", arrow(tlist(tint),tlist(tint)),
             [((l,),map(lambda n: 0-n, l))
              for _ in range(10)
              for l in [randomList()] ]),
        Task("map empty?", arrow(tlist(tlist(tint)),tlist(tboolean)),
             [((l,),map(lambda n: n == [],l))
              for _ in range(10)
              for l in [[ [] if flip() else randomList() for _ in range(randint(1,5)) ]] ]),
        
        # Task("map eq 0?", arrow(tlist(tint),tlist(tboolean)),
        #      [((l,),map(lambda n: 0 == n,l))
        #       for _ in range(10)
        #       for l in [[ randint(0,3) for _ in range(randint(4,7)) ]] ])
        
    ]

    # Learning to zip lists together
    zipBootstrap = [
        Task("zip plus", arrow(tlist(tint),tlist(tint),tlist(tint)),
             [((l1,l2),map(lambda x,y: x+y,l1,l2))
              for _ in range(5)
              for l1 in [randomList()]
              for l2 in [[ randint(0,9) for _ in range(len(l1)) ]]]),
        Task("zip minus", arrow(tlist(tint),tlist(tint),tlist(tint)),
             [((l1,l2),map(lambda x,y: x-y,l1,l2))
              for _ in range(5)
              for l1 in [randomList()]
              for l2 in [[ randint(0,9) for _ in range(len(l1)) ]]]),
    ]

    # Learning to filter
    filterBootstrap = [
        Task("remove empty lists",
             arrow(tlist(tlist(tbool)), tlist(tlist(tbool))),
             [((ls,), filter(lambda l: len(l) > 0, ls))
              for _ in range(10)
              for ls in [[[ flip() for _ in range(randint(0,3)) ]
                          for _ in range(4) ]] ]),
        # Task("remove non 0s",
        #      arrow(tlist(tint), tlist(tint)),
        #      [((xs,), filter(lambda x: x == 0, xs))
        #       for _ in range(10)
        #       for xs in [[ randint(0,3) for _ in range(5) ]] ]),
        Task("remove 0s",
             arrow(tlist(tint), tlist(tint)),
             [((xs,), filter(lambda x: x != 0, xs))
              for _ in range(10)
              for xs in [[ randint(0,3) for _ in range(5) ]] ]),
    ]

    # Learning mapi
    mapIndexBootstrap = [
        Task("Add index", arrow(tlist(tint),tlist(tint)),
             [((l,), map(lambda (i,j): i+j, enumerate(l)))
              for _ in range(10)
              for l in [randomList()] ]),
        Task("cons index", arrow(tlist(tlist(tint)),tlist(tlist(tint))),
             [((l,), map(lambda (i,j): [i]+j, enumerate(l)))
              for _ in range(10)
              for l in [randomListOfLists()] ])
        ]

    # Let's learn everything!
    if True:
        return lengthBootstrap + operationBootstrap + unfoldBootstrap + arrayBootstrap + foldBootstrap + mapBootstrap + zipBootstrap + mapIndexBootstrap + filterBootstrap            

    return [
        
        Task("increment", arrow(tint,tint),
             [((n,),n+1) for n in range(5) ]),
        Task("decrement", arrow(tint,tint),
             [((n,),n-1) for n in range(5) ]),
        Task("increment twice", arrow(tint,tint),
             [((n,),n+2) for n in range(5) ]),
        Task("increment car", arrow(tlist(tint),tint),
             [((l,),l[0] + 1)
              for _ in range(10)
              for l in [randomList()] ]),
        # Task("increment 3x", arrow(tint,tint),
        #      [((n,),n+3) for n in range(5) ]),
        Task("decrement twice", arrow(tint,tint),
             [((n,),n-2) for n in range(5) ]),
        Task("decrement car", arrow(tlist(tint),tint),
             [((l,),l[0] - 1)
              for _ in range(10)
              for l in [randomList()] ]),
        # Task("decrement 3x", arrow(tint,tint),
        #      [((n,),n-3) for n in range(5) ]),
        Task("zero?", arrow(tint,tbool),
             [((n,), n == 0) for n in range(5) ]),
        Task("zero car?", arrow(tlist(tint),tbool),
             [(([h] + l,), h == 0)
              for _ in range(5)
              for h in [0,randint(1,9)]
              for l in [randomList()] ]),
        Task("not zero?", arrow(tint,tbool),
             [((n,), n != 0) for n in range(5) ]),
        Task("not zero car?", arrow(tlist(tint),tbool),
             [(([h] + l,), h != 0)
              for _ in range(5)
              for h in [0,randint(1,9)]
              for l in [randomList()] ]),
        # # Task("multiply", arrow(tint,tint,tint),
        # #      [((x,y),x*y)
        # #       for x in range(3)
        # #       for y in range(4) ]),

        # Task("map negate", arrow(tlist(tint),tlist(tint)),
        #      [((l,),map(lambda n: -1*n,l))
        #       for _ in range(10)
        #       for l in [[ randint(-9,9) for _ in range(randint(4,7)) ]] ]),
        Task("map eq 1?", arrow(tlist(tint),tlist(tboolean)),
             [((l,),map(lambda n: 1 == n,l))
              for _ in range(10)
              for l in [[ randint(0,3) for _ in range(randint(4,7)) ]] ]),
        Task("map double", arrow(tlist(tint),tlist(tint)),
             [((l,),map(lambda n: n*2, l))
              for _ in range(10)
              for l in [randomList()] ]),
        Task("map decrement", arrow(tlist(tint),tlist(tint)),
             [((l,),map(lambda n: n-1, l))
              for _ in range(10)
              for l in [randomList()] ]),

        Task("zip plus", arrow(tlist(tint),tlist(tint),tlist(tint)),
             [((l1,l2),map(lambda x,y: x+y,l1,l2))
              for _ in range(5)
              for l1 in [randomList()]
              for l2 in [[ randint(0,9) for _ in range(len(l1)) ]]]),
        Task("zip minus", arrow(tlist(tint),tlist(tint),tlist(tint)),
             [((l1,l2),map(lambda x,y: x-y,l1,l2))
              for _ in range(5)
              for l1 in [randomList()]
              for l2 in [[ randint(0,9) for _ in range(len(l1)) ]]]),

        Task("reverse int", arrow(tlist(tint),tlist(tint)),
             [((l,),list(reversed(l)))
              for _ in range(5)
              for l in [randomList()] ]),
        Task("reverse bool", arrow(tlist(tbool),tlist(tbool)),
             [((l,),list(reversed(l)))
              for _ in range(5)
              for l in [[flip() for _ in range(randint(0,10)) ]] ]),

        
        Task("singleton", arrow(tint,tlist(tint)),
             [((n,),[n])
              for n in range(10) ]),
        Task("length bool", arrow(tlist(tbool),tint),
             [((l,), len(l))
              for _ in range(10)
              for l in [[flip() for _ in range(randint(0,10)) ]] ]),
        Task("length int", arrow(tlist(tint),tint),
             [((l,), len(l))
              for _ in range(10)
              for l in [randomList()] ]),

        # Task("range", arrow(tint,tlist(tint)),
        #      [((n,), range(n,0,-1))
        #       for n in range(10) ]),
        Task("countdown", arrow(tint,tlist(tint)),
             [((n,), range(n+1,1,-1))
              for n in range(10) ]),
        
        Task("suffixes", arrow(tlist(tint),tlist(tlist(tint))),
             [((l,), suffixes(l))
              for _ in xrange(10)
              for l in [randomList()] ]),

        # Task("repeat int", arrow(tint,tint,tlist(tint)),
        #      [((n,k), [n]*k)
        #       for k in range(5) 
        #       for n in [randint(0,9)] ]),
        # Task("repeat bool", arrow(tint,tint,tlist(tint)),
        #      [((n,k), [n]*k)
        #       for k in range(5) 
        #       for n in [flip()] ]),
        # Task("any", arrow(tlist(tbool),tbool),
        #      [((l,), reduce(lambda x,y: x and y, l, True))
        #       for sz in range(10)
        #       for l in [[random() > 0.2 for _ in range(sz) ]] ]),
        # Task("or", arrow(tboolean,tboolean,tboolean),
        #      [((a,b),a or b)
        #       for a in [True,False]
        #       for b in [True,False] ]),
        # Task("either empty?", arrow(tlist(tint),tlist(tint),tboolean),
        #      [((a,b),a == [] or b == [])
        #       for _ in xrange(10) 
        #       for a in [randomList() if flip() else []]
        #       for b in [randomList() if flip() else []] ]),

        Task("append int", arrow(tlist(tint),tlist(tint),tlist(tint)),
             [((x,y), x+y)
              for _ in range(10)
              for [x,y] in [[randomList(),randomList()]] ]),
        Task("append bool", arrow(tlist(tbool),tlist(tbool),tlist(tbool)),
             [((x,y), x+y)
              for _ in range(10)
              for [x,y] in [[randomBooleanList(),randomBooleanList()]] ]),

        Task("index int", arrow(tint, tlist(tint), tint),
             [((n,l), l[n])
              for n in range(10)
              for l in [[randint(0,9) for _ in range(randint(n+1,n + 5)) ]]]),
        Task("index bool", arrow(tint, tlist(tbool), tbool),
             [((n,l), l[n])
              for n in range(10)
              for l in [[flip() for _ in range(randint(n+1,n + 5)) ]]]),


    ]

def bonusListProblems():
    # Taken from https://www.ijcai.org/Proceedings/75/Papers/037.pdf
    # These problems might be a lot easier if we do not use numbers
    def randomList(lb = None, ub = None):
        if lb is None: lb = 2
        if ub is None: ub = 5
        return [ randint(0,5) for _ in range(randint(lb,ub)) ]
    
    bonus = [
        Task(
            "pair reverse", arrow(tlist(tint),tlist(tint)),
             [ ((x,), [ x[j + (1 if j%2 == 0 else -1)]
                        for j in range(len(x)) ])
               for _ in range(5)
               for x in [randomList(10,10)] ]
        ),
        Task(
            "duplicate each element", arrow(tlist(tint),tlist(tint)),
            [ ((x,), [ a for z in x for a in [z]*2 ])
              for _ in range(5)
              for x in [randomList(4,6)] ]
        ),
        Task(
            "reverse duplicate each element", arrow(tlist(tint),tlist(tint)),
            [ ((x,), [ a for z in reversed(x) for a in [z]*2 ])]
        ),
        ]
    return bonus        
    
        
def exportTasks():
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
    for t in make_list_bootstrap_tasks():
        print t.describe()
        print 
    # exportTasks()
