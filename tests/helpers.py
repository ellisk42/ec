import random

from dreamcoder.task import Task
from dreamcoder.type import arrow, tint


def add1():
    x = random.choice(range(10))
    return {"i": x, "o": x + 1}


def add2():
    x = random.choice(range(10))
    return {"i": x, "o": x + 1}


def create_examples(f, name):
    example = {
        "name": name,
        "data": [f() for _ in range(5)],
    }
    return example


def get_task(example):
    task = Task(
        example["name"],
        arrow(tint, tint),
        [((ex["i"],), ex["o"]) for ex in example["data"]],
    )
    return task


def get_add1_task():
    example = create_examples(add1, "add1")
    task = get_task(example)
    return task


def get_add2_task():
    example = create_examples(add2, "add2")
    task = get_task(example)
    return task