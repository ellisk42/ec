import unittest


class TestTextMain(unittest.TestCase):

    def test_imports(self):
        try:
            from dreamcoder.domains.text.main import (
                ConstantInstantiateVisitor,
                LearnedFeatureExtractor,
                competeOnOneTask,
                sygusCompetition,
                text_options,
                main
            )
        except Exception:
            self.fail('Unable to import text module')


if __name__ == '__main__':
    unittest.main()
