import os
import json
from dreamcoder.type import *
from dreamcoder.task import Task

def convert_examples(examples):
    return [
        (tuple(x), y)
        for (x, y) in examples
    ]
    
def loadRe2Dataset(task_dataset, task_dataset_dir):
    dataset_path = os.path.join(task_dataset_dir, task_dataset)
    tasks = {"train": [], "test": []}
    
    for split in ("train", "test"):
        split_path = os.path.join(dataset_path, split)
        with open(os.path.join(split_path, "tasks.json")) as f:
            task_data = json.load(f)
            tasks[split] = [
                Task(
                    name=task["name"],
                    request=arrow(tlist(tcharacter), tlist(tcharacter)),
                    examples=convert_examples(task["examples"]),
                    features=None,
                    cache=False,
                )
                for task in task_data
            ]
            for t in tasks[split]:
                t.stringConstants = []
    return tasks["train"], tasks["test"]
                
        
        
            