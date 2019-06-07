import unittest


class TestLogoScript(unittest.TestCase):

    def test_imports(self):
        try:
            from bin.logo import (
                animateSolutions,
                dreamFromGrammar,
                list_options,
                outputDreams,
                enumerateDreams,
                visualizePrimitives,
                Flatten,
                LogoFeatureCNN)
        except Exception:
            self.fail('Unable to import logo module')


if __name__ == '__main__':
    unittest.main()
