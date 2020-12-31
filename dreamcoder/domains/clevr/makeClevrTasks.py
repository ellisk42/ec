import os, json, random, copy
from dreamcoder.task import Task
from dreamcoder.type import *
from dreamcoder.domains.clevr.clevrPrimitives import *

DEFAULT_CLEVR_DATASET_DIR = "data/clevr"

QUESTIONS_DIRECTORY = 'questions'
SCENES_DIRECTORY = 'scenes'
LANGUAGE_DIRECTORY = 'language'

QUESTION_FILE_PREFIX = 'CLEVR'
TRAIN_SPLIT = 'train'
VAL_SPLIT = 'val'

def loadAllTaskAndLanguageDatasets(args):
    """
    Loads the task data for the CLEVR train and validation task datasets and converts them into typed Dreamcoder Task objects.
    
    Expects the data directory to contain both a SCENES_DIRECTORY containing all of the referenced scene objects, and a QUESTIONS_DIRECTORY containing all of the training and validation files. Unless a provided question class is 'curriculum' dataset, we expect both a training and a validation tasks file.
    
    Returns: train : [array of Task objects]
             test  : [array of Task objects]
             languageDataset: [array of string names for the task classes used that will be used to load the corresponding natural language.]
    """
    train_tasks, test_tasks = loadAllTaskDatasets(args)
    language_datasets = loadAllLanguageDatasets(args)
    return train_tasks, test_tasks, language_datasets

def loadAllLanguageDatasets(args):
    pass
    
def loadAllTaskDatasets(args):
    """
    Loads the questions for the CLEVR datasets, and converts them into DreamCoder-style tasks.
    
    Expects:
        args.curriculumDatasets: [array of question classes that will be used for a curriculum.]
        args.taskDatasets: [array of question classes that will be used for the actual testing set. (e.g. 1_compare_integer)]
        
    Returns:
        train_tasks: [array of Task objects]
        test_tasks: [array of Task objects]
    """
    question_classes_registry = buildCLEVRQuestionClassesRegistry(args)
    all_scene_data = load_all_scenes(args)
    
    all_train_tasks, all_test_tasks = [], []
    # Load the curriculum datasets, which we don't expect to have a validation set for. 
    curriculum_datasets = args.pop("curriculumDatasets")
    curriculum_tasks = buildCLEVRTasksForAllQuestionFiles(task_datasets=curriculum_datasets, question_classes_registry=question_classes_registry, all_scene_data=all_scene_data, random_seed=args["seed"], is_curriculum=True)
    all_train_tasks += curriculum_tasks[TRAIN_SPLIT]
    
    # Load the training and validation datasets.
    task_datasets = args.pop("taskDatasets")
    main_tasks = buildCLEVRTasksForAllQuestionFiles(task_datasets=task_datasets, question_classes_registry=question_classes_registry, all_scene_data=all_scene_data, random_seed=args["seed"], is_curriculum=False)
    all_train_tasks += main_tasks[TRAIN_SPLIT]
    all_test_tasks +=  main_tasks[VAL_SPLIT]

    printf(f"Loaded a total of {len(all_train_tasks)} training tasks and {len(all_test_tasks)} testing tasks for curriculum datasets: {curriculum_datasets} and main datasest: {task_datasets}")
    
    return all_train_tasks, all_test_tasks

def buildCLEVRQuestionClassesRegistry(args):
    """
    Constructs a registry containing the filepaths for all of the available training and validation datasets.
    
    Returns: question_class_to_files_dict: {
        question_class : {"train": filepath; "val" : filepath}
    }
    """
    questions_directory = os.path.join(args.taskDatasetDir, QUESTIONS_DIRECTORY)
    
    candidate_question_files = [file for file in os.listdir(questions_directory) if file.endswith('.json') and file.startswith(QUESTION_FILE_PREFIX)]
    
    question_class_to_files_dict = dict()
    for candidate_question_file in candidate_question_files:
        (filename, split, dataset_name) = get_metadata_from_question_file(questions_directory, candidate_question_file)
        if dataset_name not in question_class_to_files_dict:
            question_class_to_files_dict[dataset_name] = {
                split : filename
            }
        else:
            question_class_to_files_dict[dataset_name][split] = filename
    
    print(f"Loaded {len(question_class_to_files_dict)} CLEVR question classes: {question_class_to_files_dict.keys()}")
    return question_class_to_files_dict

def get_metadata_from_question_file(questions_directory, filename):
    """
    Returns (full_filepath, split, class) from '{prefix}_{split}_{class}.json'.
    Prefix should not have underscores.
    Class can have underscores.
    """
    split_filename = filename.split("_")
    assert split_filename[0] == QUESTION_FILE_PREFIX
    assert split_filename[1] in [TRAIN_SPLIT, VAL_SPLIT]
    split = split_filename[1] 
    dataset_name = "_".join(split_filename[2:]).split('.json')[0] # Remove the JSON
    full_filepath = os.path.join(questions_directory, filename)
    return (full_filepath, split, dataset_name)

def load_all_scenes(args):
    """
    Loads the scene file data referenced by the CLEVR tasks. Expects both a train and a validation set to be present.
    Returns: {
        "train": {index : clevr_scene_object},
        "val" :  {index : clevr_scene_object}
    }
    """
    all_scene_data = dict()
    for split in [TRAIN_SPLIT, VAL_SPLIT]:
        clevr_scene_file = f"{QUESTION_FILE_PREFIX}_{split}_{SCENES_DIRECTORY}_5000.json"
        clevr_scene_full_filepath = os.path.join(args.taskDatasetDir, SCENES_DIRECTORY, clevr_scene_file)
        with open(clevr_scene_full_filepath) as f:
            scenes = json.load(f)['scenes']    
            all_scene_data[split] = scenes
    return all_scene_data

def buildCLEVRTasksForAllQuestionFiles(task_datasets,  question_classes_registry, all_scene_data, random_seed, is_curriculum):
    """
    Builds the CLEVR tasks for a set of question files. 
    If not is_curriculum == True, expects both a train and a validation file.
    
    Returns: 
        {
            "train": [array of training task objects]
            "val" : [array of test task objects]
        }
    """    
    # Expects train and validation question splits unless it's a curriculum dataset.
    if is_curriculum: 
        splits = [TRAIN_SPLIT]
    else:
        splits = [TRAIN_SPLIT, VAL_SPLIT]
    
    generate_all = (args.question_classes_to_generate == [GENERATE_ALL_FLAG])
    tasks = defaultdict()
    for candidate_dataset in question_classes_registry:
        if candidate_dataset in task_datasets or generate_all:
            for split in splits:
                # Load the questions and construct tasks from them iteratively.
                dataset_full_filepath = question_classes_registry[candidate_dataset][split]
                with open(dataset_full_filepath) as f:
                    all_questions_for_dataset = json.load(f)["questions"]
                for question in all_questions_for_dataset:
                    t = buildClevrTask(q, scenes[split])
                    tasks[split].append(t)
                print(f"Loading dataset {candidate_dataset}: {split}: found {len(all_questions_for_dataset)} tasks.")
    return tasks
              
def loadCLEVRDataset(task_datasets, task_dataset_dir, train_scenes, test_scenes, seed, is_curriculum=False):

    """Loads tasks. If not is_curriculum, assumes there is a train and val version of each -- DEPRECATED"""
    # TODO: append the question class to the task name.
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
    # TODO: append the question class to the task name.
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
