import os, json, random, copy
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
    if type(answers[0]) == dict: return tlist(tclevrobject), True
    elif type(answers[0]) == list: return tlist(tclevrobject), True
    elif type(answers[0]) == int: return tint, False
    elif type(answers[0]) == bool: return tbool, False
    elif answers[0] in attribute_constants['color']: return tclevrcolor, False
    elif answers[0] in attribute_constants['shape']: return tclevrshape, False
    elif answers[0] in attribute_constants['size']: return tclevrsize, False
    elif answers[0] in attribute_constants['material']: return tclevrmaterial, False
    else: 
        print("Error: cannot infer return type!")
        assert False
        
def serialize_clevr_object(x, is_output=False):
    def serialize_obj(obj):
        serialized_obj = dict()
        for k in obj:
            if isinstance(obj[k], (list, tuple)):
                serialized_obj[k] = ",".join([str(v) for v in obj[k]])
            else:
                serialized_obj[k] = obj[k]
        return serialized_obj
    if is_output: # Single object list, not tuple of arguments
        return [serialize_obj(o) for o in x]
    else: return [[serialize_obj(o) for o in obj_list] 
    for obj_list in x]
    
def buildClevrMockTask(train_task, test_attr_type=None, test_count=False, test_bool=False, test_obj_list=False, test_transform=True):
    examples = []
    answers = []
    is_special, return_type = None, None
    for example in train_task.examples:
        objs = example[0][0]
        if test_attr_type is not None:
            y = objs[0][test_attr_type]
        elif test_count:
            y = len(objs)
        elif test_bool:
            y = objs[0]["color"] == "brown"
        elif test_obj_list:
            y = objs
            is_special, return_type = True, tlist(tclevrobject)
        elif test_transform:
            y = [{ k : obj[k] if k != 'color' else 'blue' for k in obj} for obj in objs]
            is_special, return_type = True, tlist(tclevrobject)
        examples.append([example[0], y])
        answers.append(y)
    if return_type is None:
        return_type, is_special = infer_return_type(answers)
    req = arrow(tlist(tclevrobject), return_type)
    print(f"Inferred task type: {req}; is_special: {is_special}")
    t = Task(name="mock", request=req,
                examples=examples, features=None, cache=False)
    t.specialSolver = 'clevrSolver'
    t.serializeSpecialInput = serialize_clevr_object
    if is_special:
        t.specialTask = ("clevrobjectlist", []) # Requires sorting the list
        t.serializeSpecialOutput = serialize_clevr_object 
    return t

def buildClevrTask(q, input_scenes):
    name = q['question'] if type(q['question']) is str else q['question'][0]
    name = f"{q['question_index']}_{name}"
    return_type, is_special = infer_return_type(q['answers'])
    request_type = arrow(tlist(tclevrobject), return_type)
    examples = build_examples(q, input_scenes)
    t = Task(name=name, request=request_type, examples=examples, features=None, cache=False)
    
    t.specialSolver = 'clevrSolver'
    t.serializeSpecialInput = serialize_clevr_object
    if is_special:
        t.specialTask = ("clevrobjectlist", []) # Requires sorting the list
        t.serializeSpecialOutput = serialize_clevr_object 
    return t
    
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