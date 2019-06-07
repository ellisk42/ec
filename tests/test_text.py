import unittest


class TestTextScript(unittest.TestCase):

    def test_imports(self):
        try:
            from bin.text import (
                ConstantInstantiateVisitor,
                LearnedFeatureExtractor,
                competeOnOneTask,
                sygusCompetition,
                text_options)
        except Exception:
            self.fail('Unable to import text module')


if __name__ == '__main__':
    unittest.main()
