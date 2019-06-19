import unittest


class TestTowerMain(unittest.TestCase):

    def test_imports(self):
        try:
            from dreamcoder.domains.tower.main import (
                Flatten,
                TowerCNN,
                tower_options,
                dreamOfTowers,
                visualizePrimitives,
                main
            )
        except Exception:
            self.fail('Unable to import tower module')


if __name__ == '__main__':
    unittest.main()
