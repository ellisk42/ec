import unittest


class TestDreaming(unittest.TestCase):

    def test_imports(self):
        try:
            from dreamcoder.dreaming import backgroundHelmholtzEnumeration
            from dreamcoder.dreaming import helmholtzEnumeration
        except Exception:
            self.fail('Unable to import from dreaming module')


if __name__ == '__main__':
    unittest.main()
