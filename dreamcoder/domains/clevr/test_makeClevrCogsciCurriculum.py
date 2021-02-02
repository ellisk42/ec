import dreamcoder.domains.clevr.makeClevrTasks as to_test
"""
Tests for makeClevrTasks.py | Author: Catherine Wong
Tests for making and loading CLEVR tasks.

All tests are manually added to a 'test_all' function.
"""

DEFAULT_CLEVR_DATASET_DIR = 'data/clevr_cogsci'

def test_loadAllOrderedCurricula():
    mock_args = {
        "easy_curriculum_categories" : ["2_localization", "1_zero_hop_no_string"],
        "hard_curriculum_categories" : ["1_compare_integer_long", "2_remove_easy", "1_single_or_easy"],
        "num_train_tasks_per_category" :[10, 5, 5],
        "num_test_tasks_per_category" : [5, 5, 5],
        "num_data_blocks_per_category" : 1,
        'taskDatasetDir' : DEFAULT_CLEVR_DATASET_DIR,
    }
    curricula = to_test.load_all_curricula(mock_args)
    for category in mock_args["hard_curriculum_categories"]:
        for datablock in range(mock_args["num_data_blocks_per_category"] + 1):
            curriculum_categories = tuple(mock_args["easy_curriculum_categories"] + [category])
            curriculum_key = (curriculum_categories, datablock)
            assert curriculum_key in curricula
            train_tasks, test_tasks = curricula[curriculum_key]
            for idx in range(len(curriculum_categories)):
                assert len(train_tasks) <= mock_args["num_train_tasks_per_category"][idx]
                assert len(train_tasks) >= 0
                assert len(test_tasks) <= mock_args["num_train_tasks_per_category"][idx]
                assert len(test_tasks) >= 0
            

def test_all():
    print("Running tests for makeClevrCogsciCurriculum....")
    test_loadAllOrderedCurricula()
    print("....all tests passed!\n")