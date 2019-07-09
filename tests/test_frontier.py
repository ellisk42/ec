import unittest

from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.program import Program
from tests.helpers import get_add1_task, get_add2_task


class TestFrontier(unittest.TestCase):

    def test_create_frontier_json(self):
        task1 = get_add1_task()
        solutions = [
            {
                "logLikelihood": 0.0,
                "program": (
                    "(lambda (car $1))"
                )
            }
        ]
        frontier = Frontier([
            FrontierEntry(program=p,
                          logLikelihood=e["logLikelihood"],
                          logPrior=0.0)
            for e in solutions
            for p in [Program.parse(e["program"])]
        ], task=task1)

        expect = {
            'request': {'constructor': '->', 'arguments': [
                {'constructor': 'int', 'arguments': []},
                {'constructor': 'int', 'arguments': []}
            ]},
            'task': 'add1', 'programs': [
                {
                    'program': '(lambda (car $1))',
                    'logLikelihood': 0.0,
                }
            ]}
        self.assertEqual(frontier.json(), expect)

    def test_create_frontier_json_empty(self):
        task1 = get_add1_task()
        frontier = Frontier([], task=task1)

        expect = {
            'request': {'constructor': '->', 'arguments': [
                {'constructor': 'int', 'arguments': []},
                {'constructor': 'int', 'arguments': []}
            ]},
            'task': 'add1', 'programs': []}
        self.assertEqual(frontier.json(), expect)


if __name__ == '__main__':
    unittest.main()
