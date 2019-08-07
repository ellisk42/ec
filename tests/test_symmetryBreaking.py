import unittest


class TestSymmetryBreaking(unittest.TestCase):

    def test_imports(self):
        try:
            from bin.symmetryBreaking import main
        except Exception:
            self.fail('Unable to import from symmetryBreaking module')


if __name__ == '__main__':
    unittest.main()
