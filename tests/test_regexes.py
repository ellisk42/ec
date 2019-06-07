import unittest


class TestRegexesScript(unittest.TestCase):

    def test_imports(self):
        try:
            from bin.regexes import (
                LearnedFeatureExtractor,
                ConstantInstantiateVisitor,
                MyJSONFeatureExtractor,
                regex_options)
        except Exception:
            self.fail('Unable to import regexes module')


if __name__ == '__main__':
    unittest.main()
