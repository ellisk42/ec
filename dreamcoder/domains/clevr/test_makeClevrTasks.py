import dreamcoder.domains.clevr.makeClevrTasks as to_test
"""
Tests for makeClevrTasks.py | Author: Catherine Wong
Tests for making and loading CLEVR tasks.

All tests are manually added to a 'test_all' function.
"""

from types import SimpleNamespace as MockArgs

DEFAULT_CLEVR_DATASET_DIR = 'data/clevr'
        
def get_all_scene_data():
    mock_args = MockArgs(taskDatasetDir=DEFAULT_CLEVR_DATASET_DIR)
    mock_args = var(mock_args)
    all_scene_data = to_test.load_all_scenes(mock_args)
    return all_scene_data

def test_buildCLEVRQuestionClassesRegistry():
    print("....running test_buildCLEVRQuestionClassesRegistry")
    mock_args = MockArgs(taskDatasetDir=DEFAULT_CLEVR_DATASET_DIR)
    mock_args = var(mock_args)
    question_class_to_files_dict = to_test.buildCLEVRQuestionClassesRegistry(mock_args)
    
    assert len(question_class_to_files_dict) == 8
    for dataset_name in question_class_to_files_dict:
        assert to_test.TRAIN_SPLIT in question_class_to_files_dict[dataset_name]
        assert to_test.VAL_SPLIT in question_class_to_files_dict[dataset_name]
    print("\n")

def test_load_all_scenes():
    print("....running test_load_all_scenes")
    mock_args = MockArgs(taskDatasetDir=DEFAULT_CLEVR_DATASET_DIR)
    mock_args = var(mock_args)
    all_scene_data = to_test.load_all_scenes(mock_args)
    
    for split in [to_test.TRAIN_SPLIT, to_test.VAL_SPLIT]:
        assert split in all_scene_data
        assert len(all_scene_data[split])
    
    print("\n")

def test_buildCLEVRTasksForAllQuestionFiles_one_dataset():
    print("....running test_buildCLEVRTasksForAllQuestionFiles_one_dataset")
    mock_args = MockArgs(taskDatasetDir=DEFAULT_CLEVR_DATASET_DIR)
    mock_args = var(mock_args)
    question_classes_registry = to_test.buildCLEVRQuestionClassesRegistry(mock_args)
    all_scene_data = get_all_scene_data() 
    
    task_datasets = ["1_one_hop"]
    
    tasks = to_test.buildCLEVRTasksForAllQuestionFiles(task_datasets=task_datasets, question_classes_registry=question_classes_registry,all_scene_data=all_scene_data,is_curriculum=False)
    
    for split in [to_test.TRAIN_SPLIT, to_test.VAL_SPLIT]:
        assert split in tasks
        assert len(tasks[split]) > 0
        for task in tasks[split]:
            assert task_datasets[0] in task.name
    print("\n")

def test_buildCLEVRTasksForAllQuestionFiles_all_datasets():
    print("....running test_buildCLEVRTasksForAllQuestionFiles_all_dataset")
    mock_args = MockArgs(taskDatasetDir=DEFAULT_CLEVR_DATASET_DIR)
    mock_args = var(mock_args)
    question_classes_registry = to_test.buildCLEVRQuestionClassesRegistry(mock_args)
    all_scene_data = get_all_scene_data() 
    
    task_datasets = ["all"]
    
    tasks = to_test.buildCLEVRTasksForAllQuestionFiles(task_datasets=task_datasets, question_classes_registry=question_classes_registry,all_scene_data=all_scene_data,is_curriculum=False)
    
    for split in [to_test.TRAIN_SPLIT, to_test.VAL_SPLIT]:
        assert split in tasks
        assert len(tasks[split]) > 0
        for task in tasks[split]:
            dataset_name = task.name.split("-")[1]
            assert dataset_name in question_classes_registry
    print("\n")
    
def test_loadAllTaskDatasets():
    mock_args = MockArgs(taskDatasetDir=DEFAULT_CLEVR_DATASET_DIR,
    curriculumDatasets=[],taskDatasets=["all"])
    mock_args = var(mock_args)
    
    question_classes_registry = to_test.buildCLEVRQuestionClassesRegistry(mock_args)
    
    all_train_tasks, all_test_tasks = to_test.loadAllTaskDatasets(mock_args)
    assert len(all_train_tasks) > 0
    assert len(all_test_tasks) > 0
    assert len(all_test_tasks) < len(all_train_tasks)
    
    for tasks in all_train_tasks, all_test_tasks:
        for task in tasks:
            dataset_name = task.name.split("-")[1]
            assert dataset_name in question_classes_registry

def test_loadAllLanguageDatasets():
    mock_args = MockArgs(taskDatasetDir=DEFAULT_CLEVR_DATASET_DIR,
    curriculumDatasets=[],taskDatasets=["all"])
    mock_args = var(mock_args)
    question_classes_registry = to_test.buildCLEVRQuestionClassesRegistry(mock_args)
    
    language_datasets = to_test.loadAllLanguageDatasets(mock_args)
    for possible_dataset in question_classes_registry:
        assert possible_dataset in language_datasets

def test_loadAllTaskAndLanguageDatasets():
    mock_args = MockArgs(taskDatasetDir=DEFAULT_CLEVR_DATASET_DIR,
    curriculumDatasets=[],taskDatasets=["all"])
    mock_args = var(mock_args)
    
    question_classes_registry = to_test.buildCLEVRQuestionClassesRegistry(mock_args)
    
    train_tasks, test_tasks, language_datasets = to_test.loadAllTaskAndLanguageDatasets(mock_args)
    
    language_datasets = to_test.loadAllLanguageDatasets(mock_args)
    for possible_dataset in question_classes_registry:
        assert possible_dataset in language_datasets
        
    for tasks in train_tasks, test_tasks:
        for task in tasks:
            dataset_name = task.name.split("-")[1]
            assert dataset_name in question_classes_registry

def assert_same_serialized_clevr_object(object, serialized_object):
    assert type(object) == type(dict())
    assert type(serialized_object) == type(dict())
    for key in object:
        if key not in ['left', 'right', 'front', 'behind']:
            assert object[key] == serialized_object[key]
        else:
            assert ",".join([str(v) for v in object[key]]) == serialized_object[key]

def assert_all_same_serialized_tasks(all_train_tasks, all_test_tasks):
    for task_set in [all_train_tasks, all_test_tasks]:
        for task in task_set:
            assert hasattr(task, "serializeSpecialInput")
            
            serialized_examples = []
            assert len(task.examples) > 0
            for xs, y in task.examples:
                if hasattr(task, "serializeSpecialInput"):
                    serialized_xs = task.serializeSpecialInput(xs)
                    for x_idx, original_x in enumerate(xs):
                        # Each x is of type list(object)
                        serialized_x = serialized_xs[x_idx]
                        assert type(original_x) == type([])
                        assert type(serialized_x) == type([])
                        assert len(original_x) != 0

                        for object_idx, object in enumerate(original_x):
                            serialized_object = serialized_x[object_idx]
                            assert_same_serialized_clevr_object(object, serialized_object)    
                
                if type(y) == type([]):
                    assert hasattr(task, "serializeSpecialOutput")
                if hasattr(task, "serializeSpecialOutput"):
                    serialized_y = task.serializeSpecialOutput(y, is_output=True)
                    # y is itself simply of type list(object)
                    for object_idx, object in enumerate(y):
                        serialized_object = serialized_y[object_idx]
                        assert_same_serialized_clevr_object(object, serialized_object)
        
def test_serialize_clevr_object_localization():
    print("....running test_serialize_clevr_object_localization")
    mock_args = MockArgs(taskDatasetDir=DEFAULT_CLEVR_DATASET_DIR,
    curriculumDatasets=[],taskDatasets=["2_localization"])
    mock_args = var(mock_args)
    
    question_classes_registry = to_test.buildCLEVRQuestionClassesRegistry(mock_args)
    
    all_train_tasks, all_test_tasks = to_test.loadAllTaskDatasets(mock_args)
    assert_all_same_serialized_tasks(all_train_tasks, all_test_tasks)
    print("\n")
    
def test_serialize_clevr_object_zero_hop():
    print("....running test_serialize_clevr_object_zero_hop")
    mock_args = MockArgs(taskDatasetDir=DEFAULT_CLEVR_DATASET_DIR,
    curriculumDatasets=[],taskDatasets=["2_zero_hop"])
    mock_args = var(mock_args)
    question_classes_registry = to_test.buildCLEVRQuestionClassesRegistry(mock_args)
    
    all_train_tasks, all_test_tasks = to_test.loadAllTaskDatasets(mock_args)
    assert_all_same_serialized_tasks(all_train_tasks, all_test_tasks)
    print("\n")
    pass

def test_serialize_clevr_object_transform():
    print("....running test_serialize_clevr_object_transform")
    mock_args = MockArgs(taskDatasetDir=DEFAULT_CLEVR_DATASET_DIR,
    curriculumDatasets=[],taskDatasets=["2_transform"])
    mock_args = var(mock_args)
    question_classes_registry = to_test.buildCLEVRQuestionClassesRegistry(mock_args)
    
    all_train_tasks, all_test_tasks = to_test.loadAllTaskDatasets(mock_args)
    assert_all_same_serialized_tasks(all_train_tasks, all_test_tasks)
    print("\n")
    pass

def test_serialize_clevr_object_all():
    print("....running test_serialize_clevr_object_transform")
    mock_args = MockArgs(taskDatasetDir=DEFAULT_CLEVR_DATASET_DIR,
    curriculumDatasets=[],taskDatasets=["all"])
    mock_args = var(mock_args)
    question_classes_registry = to_test.buildCLEVRQuestionClassesRegistry(mock_args)
    
    all_train_tasks, all_test_tasks = to_test.loadAllTaskDatasets(mock_args)
    assert_all_same_serialized_tasks(all_train_tasks, all_test_tasks)
    print("\n")
    pass
    
    
def test_all():
    print("Running tests for makeClevrTasks....")
    test_buildCLEVRQuestionClassesRegistry()
    test_load_all_scenes()
    test_buildCLEVRTasksForAllQuestionFiles_one_dataset()
    test_buildCLEVRTasksForAllQuestionFiles_all_datasets()
    test_loadAllTaskDatasets()
    test_loadAllLanguageDatasets()
    test_loadAllTaskAndLanguageDatasets()
    test_serialize_clevr_object_localization()
    test_serialize_clevr_object_zero_hop()
    test_serialize_clevr_object_transform()
    test_serialize_clevr_object_all()
    print("....all tests passed!\n")