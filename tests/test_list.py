import unittest
from collections import namedtuple

from dreamcoder.domains.list.main import train_necessary

T = namedtuple('FakeTask', 'name')


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

    def test_train_necessary(self):
        self.assertEqual(train_necessary(T('head')), True)
        self.assertEqual(train_necessary(T('add-k')), "some")
        self.assertEqual(train_necessary(T('foo')), False)


if __name__ == '__main__':
    unittest.main()
