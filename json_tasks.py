"""Load JSON tasks."""

import json

from task import RegressionTask
from type import guess_type, arrow


def hashable(v):
    """Determine whether `v` can be hashed."""
    try:
        hash(v)
    except TypeError:
        return False
    return True


def load_json_task(task, use_test=False):
    """Create a RegressionTask from the JSON representation.

    :param task: dictionary of the form {
        "name": "...",
        "train": [{"i": ..., "o": ...}],
    }
    :param use_test: if true, task must have "test" field like "train" field,
                     and it will be used alongside training data.
    :return: RegressionTask
    """
    name = task["name"]
    examples = task["train"]
    if use_test:
        examples += task["test"]
    examples = [(x["i"], x["o"]) for x in examples]

    i, o = examples[0]
    input_type = guess_type(i)
    output_type = guess_type(o)

    program_type = arrow(input_type, output_type)
    cache = hashable(i) and hashable(o)

    return RegressionTask(name, program_type,
                          [((i,), o) for i, o in examples],
                          features=examples,
                          cache=cache)


def load_json_tasks(tasks, use_test=False):
    """Create RegressionTasks from a list of the JSON task representation.

    :param tasks: list of dictionaries taking the form as described by
                  `load_json_task'.
    :param use_test: if true, test data will be used alongside training data.
    :return: list of RegressionTask.
    """
    return [load_json_task(task, use_test) for task in tasks]


def load_json_tasks_from_file(filename, use_test=False):
    """Create RegressionTasks from a JSON file.

    :param filename: JSON file, containing a list of objects taking the form as
                     described by `load_json_task'.
    :param use_test: if true, test data will be used alongside training data.
    :return: list of RegressionTask.
    """
    with open(filename) as f:
        return load_json_tasks(json.load(f), use_test)

