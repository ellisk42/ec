import os, json, random
from dreamcoder.task import Task
from dreamcoder.type import *
from dreamcoder.domains.clevr.clevrPrimitives import *

def convert_scene(input_scene):
    # Add IDs if this is a standard input scene.
    obj_ids = [idx if not "id" in obj else obj["id"] for (idx, obj) in enumerate(input_scene['objects'])]
    return [
        {   "id" : obj_ids[idx],
            "color" : obj["color"], 
            "size" : obj["size"],
            "shape" : obj["shape"],
            "material" : obj["material"],
            "left" : input_scene["relationships"]["left"][obj_ids[idx]],
            "right" : input_scene["relationships"]["right"][obj_ids[idx]],
            "front" : input_scene["relationships"]["front"][obj_ids[idx]],
            "behind" : input_scene["relationships"]["behind"][obj_ids[idx]],
        } for idx, obj in enumerate(input_scene['objects'])
    ]

def build_examples(q, input_scenes):
    examples = []
    for i, img_idx in enumerate(q['image_indices']):
        x = convert_scene(input_scenes[img_idx])
        y = q['answers'][i]
        if type(y) == dict: y = convert_scene(y)
        examples.append(([x], y))
    return examples

def infer_return_type(answers):
    if type(answers[0]) == dict: return tlist(tclevrobject)
    elif type(answers[0]) == int: return tint
    elif type(answers[0]) == bool: return tbool
    elif answers[0] in attribute_constants['color']: return tclevrcolor
    elif answers[0] in attribute_constants['shape']: return tclevrshape
    elif answers[0] in attribute_constants['size']: return tclevrsize
    elif answers[0] in attribute_constants['material']: return tclevrmaterial
    else: 
        print("Error: cannot infer return type!")
        assert False
def serialize_clevr_object(x):
    def serialize_obj(obj):
        serialized_obj = dict()
        for k in obj:
            if isinstance(obj[k], (list, tuple)):
                serialized_obj[k] = ",".join([str(v) for v in obj[k]])
            else:
                serialized_obj[k] = obj[k]
        return serialized_obj
    return [[serialize_obj(o) for o in obj_list] 
    for obj_list in x]
    
def buildClevrMockTask(train_task):
    print("Example:")
    for obj in train_task.examples[0][0][0]:
        print(f'Id: {obj["id"]}, Color: {obj["color"]}, Shape: {obj["shape"]}, Size: {obj["size"]}')
    first_obj =  train_task.examples[0][0][0][0]
    print(first_obj["left"])
    req = arrow(tlist(tclevrobject), tlist(tclevrobject))
    obj_examples = [train_task.examples[0]]
    return Task(name="mock", request=req,
                examples=obj_examples, features=None, cache=False)

def buildClevrTask(q, input_scenes):
    name = q['question'] if type(q['question']) is str else q['question'][0]
    name = f"{q['question_index']}_{name}"
    
    request_type = arrow(tlist(tclevrobject), infer_return_type(q['answers']))
    examples = build_examples(q, input_scenes)
    return Task(name=name, request=request_type, examples=examples, features=None, cache=False)
    
def loadCLEVRDataset(task_datasets, task_dataset_dir, train_scenes, test_scenes, seed, is_curriculum=False):
    """Loads tasks. If not is_curriculum, assumes there is a train and val version of each"""
    with open(os.path.join(task_dataset_dir, "scenes", train_scenes + ".json"), 'r') as f:
        train_scenes = json.load(f)['scenes']
    with open(os.path.join(task_dataset_dir, "scenes", test_scenes + ".json"), 'r') as f:
        test_scenes = json.load(f)['scenes']
    
    tasks = {"train" : [], "val" : []} if not is_curriculum else {"train" : []}
    scenes = {"train" : train_scenes, "val" : test_scenes}
    
    dataset_prefix = "CLEVR_%s_"
    for task_dataset in task_datasets:
        for split in tasks.keys():
            split_fn = (dataset_prefix % split) + task_dataset + ".json"
            dataset_fn = os.path.join(task_dataset_dir, "questions", split_fn)
            with open(dataset_fn) as f:
                qs = json.load(f)["questions"]
            for q in qs:
                t = buildClevrTask(q, scenes[split])
                tasks[split].append(t)
    if is_curriculum:
        tasks["val"] = []
    
    # Handle random shuffling
    for split in tasks:
        random.Random(seed).shuffle(tasks[split])
    return tasks["train"], tasks["val"]