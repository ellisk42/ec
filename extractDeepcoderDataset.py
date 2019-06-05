import sys
import json
import pickle as pickle

from utilities import eprint, hashable
from lib.domains.list.makeListTasks import list_features
from lib.task import Task
from type import guess_type, arrow


def extractTasks(dataset):
    for i, function in enumerate(dataset):
        name = "deep-coder #{}".format(i)
        examples = [((x["input"],), x["output"]) for x in function["examples"]]
        try:
            input_type = guess_type([i for (i,), _ in examples])
            output_type = guess_type([o for _, o in examples])
        except ValueError:
            continue
        program_type = arrow(input_type, output_type)
        features = list_features(examples)
        cache = all(hashable(x) for x in examples)
        yield Task(name, program_type, examples, features=features, cache=cache)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        eprint("must supply dataset.json argument")
    eprint("Reading dataset")
    filename = sys.argv[1]
    with open(filename) as f:
        raw = json.load(f)
    eprint("Creating tasks")
    tasks = list(extractTasks(raw))
    eprint("Got {} list tasks".format(len(tasks)))
    with open("data/list_tasks_deepcoder.pkl", "wb") as f:
        pickle.dump(tasks, f)
    eprint("Wrote deep-coder list tasks to data/list_tasks_deepcoder.pkl")
