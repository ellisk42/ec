import datetime
import os

from ec import commandlineArguments, ecIterator
from grammar import Grammar
from program import Primitive
from lib.tasks.task import Task
from type import t0, arrow, tlist, tint, tbool
from utilities import numberOfCPUs

# input/output types
intlist = arrow(tlist(tint), tlist(tint))

# primitives
def _if(c): return lambda t: lambda f: t if c else f
def _addition(x): return lambda y: x + y
def _subtraction(x): return lambda y: x - y
def _cons(x): return lambda y: [x] + y
def _car(x): return x[0]
def _cdr(x): return x[1:]
def _isEmpty(x): return x == []


if __name__ == "__main__":

    # options copied from list.py
    args = commandlineArguments(
        enumerationTimeout=10, activation='tanh',
        iterations=10, recognitionTimeout=3600,
        a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
        helmholtzRatio=0.5, structurePenalty=1.,
        CPUs=numberOfCPUs())

    timestamp = datetime.datetime.now().isoformat()
    outdir = 'experimentOutputs/demo/'
    os.makedirs(outdir, exist_ok=True)
    outprefix = outdir + timestamp
    args.update({"outputPrefix": outprefix})

    # create list of primitives copied from listPrimitives.py
    primitives = [
        # learned primitives
        Primitive("length", arrow(tlist(t0), tint), len),

        # built-ins
        Primitive("if", arrow(tbool, t0, t0, t0), _if),
        Primitive("+", arrow(tint, tint, tint), _addition),
        Primitive("-", arrow(tint, tint, tint), _subtraction),
        Primitive("empty", tlist(t0), []),
        Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), _cons),
        Primitive("car", arrow(tlist(t0), t0), _car),
        Primitive("cdr", arrow(tlist(t0), tlist(t0)), _cdr),
        Primitive("empty?", arrow(tlist(t0), tbool), _isEmpty),
    ]

    # create grammar
    grammar = Grammar.uniform(primitives)

    # construct list of tasks
    # tasks = []
    # for i in range(4):
    #     tasks.extend([
    #         Task("keep eq %s" % i,
    #              intlist,
    #              [((xs,), list(filter(lambda x: x == i, xs)))
    #               for _ in range(15)
    #               for xs in [[random.randint(0, 6) for _ in range(5)]]]),
    #         Task("remove eq %s" % i,
    #              intlist,
    #              [((xs,), list(filter(lambda x: x != i, xs)))
    #               for _ in range(15)
    #               for xs in [[random.randint(0, 6) for _ in range(5)]]]),
    #         Task("keep gt %s" % i,
    #              intlist,
    #              [((xs,), list(filter(lambda x: x > i, xs)))
    #               for _ in range(15)
    #               for xs in [[random.randint(0, 6) for _ in range(5)]]]),
    #         Task("remove gt %s" % i,
    #              intlist,
    #              [((xs,), list(filter(lambda x: not x > i, xs)))
    #               for _ in range(15)
    #               for xs in [[random.randint(0, 6) for _ in range(5)]]])
    #     ])

    # sample from list_tasks.json
    examples = [
        {"type": {"input": tlist(tint), "output": tint}, "name": "len", "examples": [
            {"i": [1, 2, 3], "o": 3},
            {"i": [0], "o": 1},
            {"i": [1, 1, 2, 1], "o": 4},
            {"i": [2, 9], "o": 2},
            {"i": [0], "o": 1},
            {"i": [10, 14, 8, 2, 12, 10, 3], "o": 7},
            {"i": [], "o": 0},
            {"i": [2, 7], "o": 2},
            {"i": [13, 11, 10, 12, 13], "o": 5},
            {"i": [15], "o": 1},
            {"i": [5, 6, 2, 8, 9], "o": 5},
            {"i": [], "o": 0},
            {"i": [3], "o": 1},
            {"i": [7, 14, 11], "o": 3},
            {"i": [15, 15, 0, 1, 3, 16], "o": 6}
        ]}
    ]
    tasks = [Task(
        item["name"],
        arrow(item["type"]["input"], item["type"]["output"]),
        [((ex["i"],), ex["o"]) for ex in item["examples"]],
    ) for item in examples]

    train = tasks

    test = [
        {"type": {"input": tlist(tint), "output": tint}, "name": "len", "examples": [
            {"i": [9], "o": 1},
            {"i": [7, 11], "o": 2},
            {"i": [12, 1, 9], "o": 3},
            {"i": [3, 14, 1, 9, 2, 1], "o": 6},
            {"i": [1, 1], "o": 2}
        ]}
    ]

    # EC iterate
    generator = ecIterator(grammar, train,
                           testingTasks=test,
                           **args)
    for i, _ in enumerate(generator):
        print('ecIterator count {}'.format(i))
