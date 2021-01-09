import os
import json
from dreamcoder.type import *
from dreamcoder.domains.re2.re2Primitives import tfullstr, tsubstr
from dreamcoder.task import Task

def convert_examples(examples, type_request):
    if type_request == "list_tcharacter":
        return [
            (tuple(x), y)
            for (x, y) in examples
        ]
    elif type_request == "tfullstr":
        return [
            (tuple(["".join(x) for x in xs]), "".join(y))
            for (xs, y) in examples
        ]
    else:
        print("Unknown type request!")
        assert False
    
def loadRe2Dataset(task_dataset, task_dataset_dir, type_request):
    dataset_path = os.path.join(task_dataset_dir, task_dataset)
    tasks = {"train": [], "test": []}
    
    if type_request == "list_tcharacter":
        request = arrow(tlist(tcharacter), tlist(tcharacter))
    elif type_request == "tfullstr":
        request = arrow(tfullstr, tfullstr)
    else: 
        print(type_request)
        assert False
    for split in ("train", "test"):
        split_path = os.path.join(dataset_path, split)
        with open(os.path.join(split_path, "tasks.json")) as f:
            task_data = json.load(f)
            tasks[split] = [
                Task(
                    name=task["name"],
                    request=request,
                    examples=convert_examples(task["examples"], type_request),
                    features=None,
                    cache=False,
                )
                for task in task_data
            ]
            for t in tasks[split]:
                t.stringConstants = []
    return tasks["train"], tasks["test"]
                
def buildRE2MockTask(train_task):
    """Builds a mock task for testing"""
    example = train_task.examples[0]
    example = (example[0], example[0][0])
    print("Mock Example:")
    print(example)
    req = arrow(tfullstr, tfullstr)
    return Task(name="mock", request=req,
                examples=[example, example], features=None, cache=False)
            