import unittest


class TestRecognition(unittest.TestCase):

    def test_imports(self):
        try:
            from dreamcoder.recognition import RecognitionModel
        except Exception:
            self.fail('Unable to import from recognition module')


if __name__ == '__main__':
    unittest.main()
