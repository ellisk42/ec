import random
import unittest
from unittest import mock

from dreamcoder.enumeration import multicoreEnumeration
from dreamcoder.frontier import Frontier
from dreamcoder.grammar import Grammar
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint


def add1():
    x = random.choice(range(10))
    return {"i": x, "o": x + 1}


def add2():
    x = random.choice(range(10))
    return {"i": x, "o": x + 1}


def create_examples(f, name):
    example = {
        "name": name,
        "data": [f() for _ in range(5)],
    }
    return example


def get_task(example):
    task = Task(
        example["name"],
        arrow(tint, tint),
        [((ex["i"],), ex["o"]) for ex in example["data"]],
    )
    return task


def get_add1_task():
    example = create_examples(add1, "add1")
    task = get_task(example)
    return task


def get_add2_task():
    example = create_examples(add2, "add2")
    task = get_task(example)
    return task


class TestEnumerationMain(unittest.TestCase):

    def test_multicore_enumeration_no_tasks(self):
        grammar = Grammar.uniform([])
        tasks = []
        frontiers, best_search_time = multicoreEnumeration(grammar, tasks)
        self.assertEqual(frontiers, [])
        self.assertEqual(best_search_time, {})

    @mock.patch('dreamcoder.enumeration.subprocess')
    def test_multicore_enumeration_single_task(self, mock_subprocess):
        mock_process = mock.MagicMock()
        response = '{"add1": []}'.encode('utf-8')
        mock_process.communicate.return_value = (response, None)
        mock_subprocess.Popen.return_value = mock_process
        grammar = Grammar.uniform([])
        task = get_add1_task()
        tasks = [task]
        frontiers, best_search_time = multicoreEnumeration(
            grammar, tasks, maximumFrontier=1, enumerationTimeout=1)
        self.assertIsInstance(frontiers, list)
        self.assertEqual(len(frontiers), 1)
        actual_frontier = frontiers[0]
        expect_frontier = Frontier([], task)
        self.assertEqual(actual_frontier.entries, expect_frontier.entries)
        self.assertEqual(actual_frontier.task, expect_frontier.task)
        self.assertIsInstance(best_search_time, dict)
        self.assertEqual([t.name for t in best_search_time.keys()], ['add1'])

    @mock.patch('dreamcoder.enumeration.subprocess')
    def test_multicore_enumeration_multiple_tasks(self, mock_subprocess):
        mock_process = mock.MagicMock()
        response = '{"add1": [], "add2": []}'.encode('utf-8')
        mock_process.communicate.return_value = (response, None)
        mock_subprocess.Popen.return_value = mock_process
        grammar = Grammar.uniform([])
        task1 = get_add1_task()
        task2 = get_add2_task()
        tasks = [task1, task2]
        frontiers, best_search_time = multicoreEnumeration(
            grammar, tasks, maximumFrontier=1, enumerationTimeout=1)
        self.assertIsInstance(frontiers, list)
        self.assertEqual(len(frontiers), 2)
        actual_frontier1 = frontiers[0]
        expect_frontier1 = Frontier([], task1)
        self.assertEqual(actual_frontier1.entries, expect_frontier1.entries)
        self.assertEqual(actual_frontier1.task, expect_frontier1.task)
        actual_frontier2 = frontiers[1]
        expect_frontier2 = Frontier([], task2)
        self.assertEqual(actual_frontier2.entries, expect_frontier2.entries)
        self.assertEqual(actual_frontier2.task, expect_frontier2.task)
        self.assertIsInstance(best_search_time, dict)
        self.assertEqual([t.name for t in best_search_time.keys()], ['add1', 'add2'])

    @mock.patch('dreamcoder.enumeration.subprocess')
    def test_multicore_enumeration_invalid_response_error(self, mock_subprocess):
        mock_process = mock.MagicMock()
        response = '{"OOPS": []}'.encode('utf-8')
        mock_process.communicate.return_value = (response, None)
        mock_subprocess.Popen.return_value = mock_process
        grammar = Grammar.uniform([])
        task = get_add1_task()
        tasks = [task]
        with self.assertRaises(AssertionError):
            multicoreEnumeration(
                grammar, tasks, maximumFrontier=1, enumerationTimeout=1)


if __name__ == '__main__':
    unittest.main()
