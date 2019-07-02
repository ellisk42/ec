"""
This module includes more list tasks from the following master list:
https://docs.google.com/document/d/1D99neDlUYXm1v4-5pQVsjh8V2mWKHTCArQ7Q9u9ea8Q/edit
"""
from abc import ABC, abstractmethod
import json
import os
import random

from dreamcoder.utilities import get_data_dir

JSON_FILE = os.path.join(get_data_dir(), "more_list_tasks.json")


class SkipExample(Exception):
    """ Raise to skip an example for a given input. """


class TaskGenerator(ABC):
    """
    A TaskGenerator must have the following class attributes:
      - name
      - input_type
      - output_type

    A TaskGenerator must also implement the following functions:
      - func(inputs)
      - make_examples()

    Example task:

        MyTaskGenerator(TaskGenerator):
            name = None
            input_type = None
            output_type = None

            def func(self, x):
                x0 = x[0]
                return [x0 for _ in range(x0)]

            def make_examples(self):
                ...

    Example task format:

        [{
            "type": {"input": "list-of-int", "output": "list-of-int"},
            "name": "add-k with k=0",
            "examples": [
                {"i": [], "o": []},
                {"i": [1, 7, 1, 10, 1], "o": [1, 7, 1, 10, 1]},
                {"i": [2, 14], "o": [2, 14]}
        ]}

    """
    name = None
    input_type = None
    output_type = None
    num_examples = 20

    def __init__(self):
        assert self.name is not None
        assert self.input_type is not None
        assert self.output_type is not None
        self.examples = self.make_examples()

    @abstractmethod
    def func(self, inputs):
        """
        Function to be applied to inputs, generating outputs.
        """
        pass

    @abstractmethod
    def make_examples(self):
        pass

    def example(self, inputs):
        return {'i': inputs, 'o': self.func(inputs)}

    @staticmethod
    def _to_json(name, input_type, output_type, examples):
        return {
            'name': name,
            'type': {'input': input_type, 'output': output_type},
            'examples': examples
        }

    def json(self):
        return self._to_json(self.name, self.input_type, self.output_type, self.examples)


class ShuffledRangeTask(TaskGenerator, ABC):

    def make_examples(self):
        examples = [self.example([n]) for n in range(self.num_examples)]
        random.shuffle(examples)
        return examples


class RandomListTask(TaskGenerator, ABC):
    min_val = 0
    max_val = 9
    min_len = 2
    max_len = 7

    def random_list(self):
        list_range = range(random.randint(self.min_len, self.max_len))
        return [random.randint(self.min_val, self.max_val) for _ in list_range]

    def make_examples(self):
        created = 0
        examples = []
        while created < self.num_examples:
            try:
                examples.append(self.example(self.random_list()))
            except SkipExample:
                continue
            else:
                created += 1
        return examples


class RepeatN(ShuffledRangeTask):
    """
    Routine #1 from master list.

    Examples:

        (2) -> (2 2)
        (5) -> (5 5 5 5 5)
    """
    name = 'repeat_n_n_times'
    input_type = 'list-of-int'
    output_type = 'list-of-int'

    def func(self, x):
        x0 = x[0]
        return [x0 for _ in range(x0)]


class CountDown(ShuffledRangeTask):
    """
    Routine #2 from master list.

    Examples:

        (2) - (2 1)
        (5) - (5 4 3 2 1)

    """
    name = 'count_down_from_n'
    input_type = 'list-of-int'
    output_type = 'list-of-int'

    def func(self, x):
        return list(reversed([n for n in range(1, x[0] + 1)]))


class LastElement(RandomListTask):
    """
    Routine #3 from master list.

    Examples:

        (6 4 9 1 4) - 4
        (7 3 3 2) - 2
        (8 1) - 1

    """
    name = 'last_element_of_list'
    input_type = 'list-of-int'
    output_type = 'int'

    def func(self, x):
        return x[-1]


class HeadthElement(RandomListTask):
    """
    Routine #4 from master list.

    Examples:

        (3 9 2 1 8 8) - 1
        (2 7 9 1) - 9
        (6 3 1 8 6 9 2 7) - 2
        (4 9 5 2 2 3 9) - 2


    """
    name = 'headth_element_of_tail'
    input_type = 'list-of-int'
    output_type = 'int'

    def func(self, x):
        head = x[0]
        if head - 1 < 0:
            raise SkipExample
        tail = x[1:]
        try:
            return tail[head - 1]
        except IndexError:
            raise SkipExample


class CountHead(RandomListTask):
    """
    Routine #5 from master list.

    Examples:

        (9 2 6 4 9 1 9 9 3) - 3
        (3 1 7 3 9 1 3) - 2
        (6 7 1 2 9 1) - 0

    """
    name = 'count_head_in_tail'
    input_type = 'list-of-int'
    output_type = 'int'

    def func(self, x):
        head = x[0]
        tail = x[1:]
        return sum(1 for n in tail if n == head)


def create_more_list_tasks():
    tasks = [
        RepeatN(),
        CountDown(),
        LastElement(),
        HeadthElement(),
        CountHead(),
    ]

    data = []
    for t in tasks:
        data.append(t.json())

    with open(JSON_FILE, 'w') as f:
        json.dump(data, f)

    num_tasks = len(tasks)
    print(f'wrote {num_tasks} tasks to: {JSON_FILE}')


if __name__ == '__main__':
    create_more_list_tasks()
