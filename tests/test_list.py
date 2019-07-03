import unittest


class TestListMain(unittest.TestCase):

    def test_imports(self):
        try:
            from dreamcoder.domains.list.main import (
                retrieveJSONTasks,
                list_features,
                isListFunction,
                isIntFunction,
                train_necessary,
                list_options,
                LearnedFeatureExtractor,
                main
            )
        except Exception:
            self.fail('Unable to import list module')


if __name__ == '__main__':
    unittest.main()
