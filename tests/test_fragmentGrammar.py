import unittest


class TestFragmentGrammar(unittest.TestCase):

    def test_imports(self):
        try:
            from dreamcoder.fragmentGrammar import FragmentGrammar
        except Exception:
            self.fail('Unable to import from fragmentGrammar module')


if __name__ == '__main__':
    unittest.main()
