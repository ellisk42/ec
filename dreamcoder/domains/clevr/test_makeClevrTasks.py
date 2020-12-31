import dreamcoder.domains.clevr.makeClevrTasks as to_test

from types import SimpleNamespace as MockArgs

DEFAULT_CLEVR_DATASET_DIR = 'data/clevr'

def test_buildCLEVRQuestionClassesRegistry():
    print("....running test_buildCLEVRQuestionClassesRegistry")
    mock_args = MockArgs(taskDatasetDir=DEFAULT_CLEVR_DATASET_DIR)
    
    question_class_to_files_dict = to_test.buildCLEVRQuestionClassesRegistry(mock_args)
    
    assert len(question_class_to_files_dict) == 8
    for dataset_name in question_class_to_files_dict:
        assert to_test.TRAIN_SPLIT in question_class_to_files_dict[dataset_name]
        assert to_test.VAL_SPLIT in question_class_to_files_dict[dataset_name]
    print("\n")

def test_load_all_scenes():
    print("....running test_load_all_scenes")
    mock_args = MockArgs(taskDatasetDir=DEFAULT_CLEVR_DATASET_DIR)
    all_scene_data = to_test.load_all_scenes(mock_args)
    
    for split in [to_test.TRAIN_SPLIT, to_test.VAL_SPLIT]:
        assert split in all_scene_data
        assert len(all_scene_data[split])
    
    print("\n")

def test_buildCLEVRTasksForAllQuestionFiles_one_dataset():
    mock_args = MockArgs(taskDatasetDir=DEFAULT_CLEVR_DATASET_DIR)
    question_classes_registry = to_test.buildCLEVRQuestionClassesRegistry(mock_args)

def test_buildCLEVRTasksForAllQuestionFiles_all_datasets():
    pass
    
def test_all():
    
    print("Running tests for makeClevrTasks....")
    test_buildCLEVRQuestionClassesRegistry()
    test_load_all_scenes()