import unittest


class TestListScript(unittest.TestCase):

    def test_imports(self):
        try:
            from bin.list import (
                retrieveJSONTasks,
                list_features,
                isListFunction,
                isIntFunction,
                train_necessary,
                list_options,
                LearnedFeatureExtractor)
        except Exception:
            self.fail('Unable to import list module')


if __name__ == '__main__':
    unittest.main()
