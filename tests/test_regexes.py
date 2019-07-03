import unittest


class TestRegexesMain(unittest.TestCase):

    def test_imports(self):
        try:
            from dreamcoder.domains.regex.main import (
                LearnedFeatureExtractor,
                ConstantInstantiateVisitor,
                MyJSONFeatureExtractor,
                regex_options,
                main
            )
        except Exception:
            self.fail('Unable to import regexes module')


if __name__ == '__main__':
    unittest.main()
