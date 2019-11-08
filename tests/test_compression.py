import unittest


class TestCompression(unittest.TestCase):

    def test_imports(self):
        try:
            from dreamcoder.compression import induceGrammar, ocamlInduce
        except Exception:
            self.fail('Unable to import from compression module')


if __name__ == '__main__':
    unittest.main()
