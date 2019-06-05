"""
Usage:

    singularity exec container.img python demo2.py -t 20 -RS 10
"""

import datetime
import os
import random

from lib.ec import commandlineArguments, ecIterator
from grammar import Grammar
from program import Primitive
from lib.task import Task
from type import arrow, tint
from utilities import numberOfCPUs

# Primitives
def _incr(x): return lambda x: x + 1


def addN(n):
    x = random.choice(range(500))
    return {"i": x, "o": x + n}


def get_tint_task(item):
    return Task(
        item["name"],
        arrow(tint, tint),
        [((ex["i"],), ex["o"]) for ex in item["examples"]],
    )


if __name__ == "__main__":

    # Options more or less copied from list.py

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

    # Create list of primitives

    primitives = [
        Primitive("incr", arrow(tint, tint), _incr),
    ]

    # Create grammar

    grammar = Grammar.uniform(primitives)

    def add1(): return addN(1)
    def add2(): return addN(2)
    def add3(): return addN(3)
    def add5(): return addN(5)
    def add8(): return addN(8)
    def add13(): return addN(13)
    def add21(): return addN(21)

    add1_examples = [add1() for _ in range(500)]
    add2_examples = [add2() for _ in range(500)]
    add3_examples = [add3() for _ in range(500)]
    add5_examples = [add5() for _ in range(500)]
    add8_examples = [add8() for _ in range(500)]
    add13_examples = [add13() for _ in range(500)]
    add21_examples = [add21() for _ in range(500)]

    # Training data

    training_examples = [
        {"name": "add1", "examples": add1_examples},
        {"name": "add2", "examples": add2_examples},
        {"name": "add3", "examples": add3_examples},
        {"name": "add5", "examples": add5_examples},
        {"name": "add8", "examples": add8_examples},
        {"name": "add13", "examples": add13_examples},
        {"name": "add21", "examples": add21_examples},
    ]
    training_tasks = [get_tint_task(item) for item in training_examples]

    # Testing data

    def add9(): return addN(9)
    def add19(): return addN(19)

    add9_examples = [add9() for _ in range(500)]
    add19_examples = [add19() for _ in range(500)]

    testing_examples = [
        {"name": "add9", "examples": add9_examples},
        {"name": "add19", "examples": add9_examples},
    ]
    testing_tasks = [get_tint_task(item) for item in testing_examples]

    # EC iterate

    generator = ecIterator(grammar,
                           training_tasks,
                           testingTasks=testing_tasks,
                           **args)
    for i, _ in enumerate(generator):
        print('ecIterator count {}'.format(i))
