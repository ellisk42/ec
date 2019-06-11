import unittest


class TestTowerScript(unittest.TestCase):

    def test_imports(self):
        try:
            from bin.tower import (
                Flatten,
                TowerCNN,
                tower_options,
                dreamOfTowers,
                visualizePrimitives)
        except Exception:
            self.fail('Unable to import tower module')


if __name__ == '__main__':
    unittest.main()
