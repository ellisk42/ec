"""
Tests for clevrRecognition.py | Author: Catherine Wong
Test for the neural recognition model specific to the CLEVR symbolic scene graph dataset.

All tests are manually added to a 'test_all' function.

The tests are based on the recognition model(s) in clevrRecognition.py.
"""
import dreamcoder.domains.clevr.clevrRecognition as to_test

import dreamcoder.domains.clevr.makeClevrTasks as makeClevrTasks
import dreamcoder.domains.clevr.test_clevrPrimitives as test_clevrPrimitives
import dreamcoder.domains.clevr.clevrPrimitives as clevrPrimitives
from dreamcoder.program import Primitive, Program
from types import SimpleNamespace as MockArgs

DEFAULT_CLEVR_DATASET_DIR = 'data/clevr'
DATASET_CACHE = dict()

def get_train_test_task_datasets(task_dataset):
    mock_args = MockArgs(taskDatasetDir=DEFAULT_CLEVR_DATASET_DIR,
    curriculumDatasets=[],taskDatasets=[task_dataset])
    mock_args = vars(mock_args)
    if task_dataset not in DATASET_CACHE:
        all_train_tasks, all_test_tasks = makeClevrTasks.loadAllTaskDatasets(mock_args)
        DATASET_CACHE[task_dataset] = (all_train_tasks, all_test_tasks)
    else:
        (all_train_tasks, all_test_tasks) = DATASET_CACHE[task_dataset]
    return (all_train_tasks, all_test_tasks)
    
def get_default_clevr_feature_extractor_recurrent_feature_extractor(train_tasks, test_tasks):
    """Utility function to instantiate a basic feature extractor with tasks."""
    feature_extractor = to_test.ClevrFeatureExtractor(
        tasks=train_tasks,
        testingTasks=test_tasks,
        cuda=False
    )
    return feature_extractor

def assert_is_list_of_strings_in_lexicon(test_list, lexicon):
    assert type(test_list) == type([])
    for item in test_list:
        assert type(item) == type("")
        assert item in lexicon
        
def test_clevr_feature_extractor_recurrent_tokenizer():
    """Tests the tokenization on the CLEVR feature extractor, which takes a task and renders a flat set of tokens to be processed recurrently."""
    (all_train_tasks, all_test_tasks) = get_train_test_task_datasets(task_dataset="all")
    feature_extractor = get_default_clevr_feature_extractor_recurrent_feature_extractor(all_train_tasks, all_test_tasks)
    for task in all_train_tasks:
        tokenized = feature_extractor.tokenize(task)
        assert len(tokenized) <= feature_extractor.max_examples
        for (xs, y) in tokenized:
            y_len = len(y)
            xs_len = sum([len(x) for x in xs])
            assert_is_list_of_strings_in_lexicon(y, feature_extractor.lexicon)
            for x in xs:
                assert_is_list_of_strings_in_lexicon(y, feature_extractor.lexicon)
            assert y_len + xs_len <= feature_extractor.max_examples_length
            
def test_clevr_feature_extractor_recurrent_taskOfProgram_success():
    """Tests the task of program functionality of the recurrent feature extractor, which calls out to the RecurrentFeatureExtractor processing."""
    (all_train_tasks, all_test_tasks) = get_train_test_task_datasets(task_dataset="all")
    feature_extractor = get_default_clevr_feature_extractor_recurrent_feature_extractor(all_train_tasks, all_test_tasks)
    
    clevrPrimitives.clevr_original_v1_primitives()
    transform_task, raw_program_original, raw_program_bootstrap = test_clevrPrimitives.test_default_transform()
    program = Program.parse(raw_program_bootstrap)
    task_type = transform_task.request
    original_return_type = makeClevrTasks.infer_return_type([transform_task.examples[0][-1]])
    
    generated_task = feature_extractor.taskOfProgram(program, task_type)
    assert generated_task.name == 'Helmholtz'
    assert generated_task.request == transform_task.request
    assert len(generated_task.examples) > 0
    assert len(generated_task.examples) <= feature_extractor.max_examples
    assert makeClevrTasks.infer_return_type([generated_task.examples[0][-1]]) == original_return_type
    

def test_clevr_feature_extractor_recurrent_taskOfProgram_malformed():
    """Tests the task of program functionality of the recurrent feature extractor, which calls out to the RecurrentFeatureExtractor processing, when the program itself is malformed."""
    (all_train_tasks, all_test_tasks) = get_train_test_task_datasets(task_dataset="all")
    feature_extractor = get_default_clevr_feature_extractor_recurrent_feature_extractor(all_train_tasks, all_test_tasks)
    
    (all_train_tasks, all_test_tasks) = get_train_test_task_datasets(task_dataset="all")
    feature_extractor = get_default_clevr_feature_extractor_recurrent_feature_extractor(all_train_tasks, all_test_tasks)
    
    clevrPrimitives.clevr_original_v1_primitives()
    test_task, _ = test_clevrPrimitives.test_fold_malformed()
    raw_degenerate = "(lambda (clevr_empty))"
    program = Program.parse(raw_degenerate)
    task_type = test_task.request
    original_return_type = makeClevrTasks.infer_return_type([test_task.examples[0][-1]])
    
    generated_task = feature_extractor.taskOfProgram(program, task_type)
    assert generated_task is None
    
    

def test_clevr_feature_extractor_recurrent_forward_regular_task():
    """Tests the complete CLEVR feature extractor forward function for a given task.
    Test the time and memory usages of a single pass.
    """
    pass

def integration_test_clevr_feature_extractor_recurrent_recognition_gradient_step():
    """Integration test for the behavior of a single gradient step with the CLEVR feature extractor as the example encoder."""
    pass

def integration_test_clevr_feature_extractor_recurrent_recognition_gradient_step():
    """Tests the complete CLEVR feature extractor forward function for a given task that is used for enumeration.
    """
    pass

def run_test(test_fn):
    """Utility function for running tests"""
    print(f"Running {test_fn.__name__}...")
    test_fn()
    print("\n")

def test_all():
    print("Running tests for clevrRecognition...")
    # run_test(test_clevr_feature_extractor_recurrent_tokenizer)
    # run_test(test_clevr_feature_extractor_recurrent_taskOfProgram_success)
    run_test(test_clevr_feature_extractor_recurrent_taskOfProgram_malformed)
    print(".....finished running all tests!")