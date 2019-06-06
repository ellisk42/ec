"""
Demo #3 involving a new primitive (incr2).

New primitive requires rebuild of OCaml binaries:

    cd ec
    ./container.img
    make clean
    make

Training Usage:

    singularity exec container.img python demo3.py -t 10

Usage w/out bottom-up neural recognition model:

    singularity exec container.img python demo3.py -t 10 -g

Usage with incredibly low DSL acceptance threshold:

    singularity exec container.img python demo3.py -t 1 -g -l -1000000 --aic -1000000

Testing Usage:

    singularity exec container.img python demo3.py -t 2 --testingTimeout 2

"""

import datetime
import os
import random

import binutil

from lib.ec import commandlineArguments, ecIterator
from lib.grammar import Grammar
from lib.program import Primitive
from lib.task import Task
from lib.type import arrow, tint
from lib.utilities import numberOfCPUs

# Primitives
def _incr(x): return lambda x: x + 1
def _incr2(x): return lambda x: x + 2


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
        Primitive("incr2", arrow(tint, tint), _incr2),
    ]

    # Create grammar

    grammar = Grammar.uniform(primitives)

    def add1(): return addN(1)
    def add2(): return addN(2)
    def add3(): return addN(3)

    # Training data

    training_examples = [
        {"name": "add1", "examples": [add1() for _ in range(5000)]},
        {"name": "add2", "examples": [add2() for _ in range(5000)]},
        {"name": "add3", "examples": [add3() for _ in range(5000)]},
    ]
    training_tasks = [get_tint_task(item) for item in training_examples]

    # Testing data

    def add4(): return addN(4)

    testing_examples = [
        {"name": "add4", "examples": [add4() for _ in range(500)]},
    ]
    testing_tasks = [get_tint_task(item) for item in testing_examples]

    # EC iterate

    generator = ecIterator(grammar,
                           training_tasks,
                           testingTasks=testing_tasks,
                           **args)
    for i, _ in enumerate(generator):
        print('ecIterator count {}'.format(i))
