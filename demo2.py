"""
Usage:

    singularity exec container.img python demo.py -t 20 -RS 10
"""

import datetime
import os
import random

from ec import commandlineArguments, ecIterator
from grammar import Grammar
from program import Primitive
from task import Task
from type import t0, arrow, tint
from utilities import numberOfCPUs

# input/output types

# primitives
def _and(x): return lambda y: x and y
def _eq(x): return lambda y: x == y
def _gt(x): return lambda y: x > y
def _if(x): return lambda t: lambda f: t if x else f
def _not(x): return not x
def _or(x): return lambda y: x or y

def _addition(x): return lambda y: x + y
def _subtraction(x): return lambda y: x - y
def _addition(x): return lambda y: x + y
def _subtraction(x): return lambda y: x - y

def _plus1(x): return lambda x: x + 1
def _plus2(x): return lambda x: x + 2


def plusN(n):
    x = random.choice(range(500))
    return {"i": x, "o": x + n}


def get_tint_task(item):
    return Task(
        item["name"],
        arrow(tint, tint),
        [((ex["i"],), ex["o"]) for ex in item["examples"]],
    )


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
        Primitive("+1", arrow(tint, tint, tint), _plus1),
        Primitive("+2", arrow(tint, tint, tint), _plus2),
    ]

    # create grammar
    grammar = Grammar.uniform(primitives)

    def plus1(): return plusN(1)
    def plus2(): return plusN(2)
    def plus3(): return plusN(3)
    def plus5(): return plusN(5)
    def plus8(): return plusN(8)
    def plus13(): return plusN(13)
    def plus21(): return plusN(21)

    plus1_examples = [plus1() for _ in range(500)]
    plus2_examples = [plus2() for _ in range(500)]
    plus3_examples = [plus3() for _ in range(500)]
    plus5_examples = [plus5() for _ in range(500)]
    plus8_examples = [plus8() for _ in range(500)]
    plus13_examples = [plus13() for _ in range(500)]
    plus21_examples = [plus21() for _ in range(500)]

    training_examples = [
        {"name": "plus1", "examples": plus1_examples},
        {"name": "plus2", "examples": plus2_examples},
        {"name": "plus3", "examples": plus3_examples},
        {"name": "plus5", "examples": plus5_examples},
        {"name": "plus8", "examples": plus8_examples},
        {"name": "plus13", "examples": plus13_examples},
        {"name": "plus21", "examples": plus21_examples},
    ]
    training_tasks = [get_tint_task(item) for item in training_examples]

    def plus9(): return plusN(9)
    def plus19(): return plusN(19)

    plus9_examples = [plus9() for _ in range(500)]
    plus19_examples = [plus19() for _ in range(500)]

    testing_examples = [
        {"name": "plus9", "examples": plus9_examples},
        {"name": "plus19", "examples": plus9_examples},
    ]
    testing_tasks = [get_tint_task(item) for item in testing_examples]

    # EC iterate
    generator = ecIterator(grammar,
                           training_tasks,
                           testingTasks=testing_tasks,
                           **args)
    for i, _ in enumerate(generator):
        print('ecIterator count {}'.format(i))
