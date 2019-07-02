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

Integer = 'int'
ListOfInts = 'list-of-int'


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
    """
    A ShuffledRangeTask has the following optional knobs as class attributes:
      - num_examples (int): number of examples to create
    """
    num_examples = 20

    def make_examples(self):
        examples = [self.example([n]) for n in range(self.num_examples)]
        random.shuffle(examples)
        return examples


class RandomListTask(TaskGenerator, ABC):
    """
    A RandomListTask has the following optional knobs as class attributes:
      - min_val (int): minimum value in random list created
      - max_val (int): maximum value in random list created
      - min_len (int): minimum length of random list created
      - max_len (int): maximum length of random list created
      - num_examples (int): number of random lists to create
    """
    min_val = 0
    max_val = 9
    min_len = 2
    max_len = 7
    num_examples = 20

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

        (2) - (2 2)
        (5) - (5 5 5 5 5)
    """
    name = 'repeat_n_n_times'
    input_type = ListOfInts
    output_type = ListOfInts

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
    input_type = ListOfInts
    output_type = ListOfInts

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
    input_type = ListOfInts
    output_type = Integer

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
    input_type = ListOfInts
    output_type = Integer

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
    input_type = ListOfInts
    output_type = Integer
    num_examples = 100  # fails under 20 examples, trying a higher limit

    def func(self, x):
        head = x[0]
        tail = x[1:]
        return sum(1 for n in tail if n == head)


# TODO: what about zeroes?
class FlattenMapRange(RandomListTask):
    """
    Routine #6 from master list.

    Examples:

        (2 5 4) - (1 2 1 2 3 4 5 1 2 3 4)
        (3 2) - (1 2 3 1 2)

    """
    name = 'flatten_map_range'
    input_type = ListOfInts
    output_type = ListOfInts
    num_examples = 100  # fails under 20 examples

    def func(self, x):
        return [i for j in list(map(lambda n: range(1, n + 1), x)) for i in j]


class FlattenMapRangeReversed(RandomListTask):
    """
    Routine #6a from master list.

    Examples:

        (2 5 4) - (2 1 5 4 3 2 1 4 3 2 1)
        (3 2) - (3 2 1 2 1)

    """
    name = 'flatten_map_range_reversed'
    input_type = ListOfInts
    output_type = ListOfInts
    num_examples = 100  # fails under 20 examples

    def func(self, x):
        return [i for j in list(map(lambda n: reversed(range(1, n + 1)), x)) for i in j]


class FlattenMapRangeSeries(RandomListTask):
    """
    Routine #6b from master list.

    Examples:

        (4 8 1 3) - (4 5 6 7 8 7 6 5 4 3 2 1 2 3)
        (9 7 7 7 3 4 4 2 6) - (9 8 7 7 7 6 5 4 3 4 4 3 2 3 4 5 6)
        (3 2 1 2) - (3 2 1 2)
        (4 1 2 5) - (4 3 2 1 2 3 4 5)

    """
    name = 'flatten_map_range_series'
    input_type = ListOfInts
    output_type = ListOfInts
    num_examples = 100  # fails under 20 examples

    def func(self, x):
        if len(x) <= 1:
            return x
        pairs = [(x[a], x[a + 1]) for a in range(len(x) - 1)]

        output = []

        for index, p in enumerate(pairs):
            p0, p1 = p
            if p0 > p1:
                l = range(p0, (p1 - 1), -1)
            elif p0 < p1:
                l = range(p0, (p1 + 1))
            else:
                l = [p0]
            if index != 0 and len(l) > 1:
                # avoid duplicating the same number
                l = l[1:]
            output.extend(l)

        return output


class FlattenMapRangeHead(RandomListTask):
    """
    Routine #6c from master list.

    Examples:

        (4 8 1 3) - (4 5 6 7 8 1 3)
        (5 1 9 7 7 3 4 4 2 6) - (5 1 5 6 7 8 9 5 6 7 5 6 7 3 4 4 2 5 6)
        (1 3 6 2) - (1 2 3 1 2 3 4 5 6 1 2)
        (3 1 2 9) - (3 1 2 3 4 5 6 7 8 9)
        (4 8 1 7) - (4 5 6 7 8 1 4 5 6 7)
        (3 1 9 7 3 4 4 2 6) - (3 1 3 4 5 6 7 8 9 3 4 5 6 7 3 3 4 3 4 2 3 4 5 6)
        (7 1 2 9) - (7 1 2 7 8 9)


    """
    name = 'flatten_map_range_head'
    input_type = ListOfInts
    output_type = ListOfInts
    num_examples = 100  # fails under 20 examples

    def func(self, x):
        if len(x) <= 1:
            return x

        head = x[0]
        tail = x[1:]
        output = []

        for index, val in enumerate(tail):
            if head < val:
                l = range(head, (val + 1))
            else:
                if index == 0:
                    l = [head, val]
                else:
                    l = [val]
            output.extend(l)

        return output


class Minus2Series(RandomListTask):
    """
    Routine #7 from master list.

    Examples:

        (6) - (6 4 2)
        (9) - (9 7 5 3 1)
        (18) - (18 16 14 12 10 8 6 4 2)

    """
    name = 'minus_2_series'
    input_type = ListOfInts
    output_type = ListOfInts

    def func(self, x):
        n = x[0]
        if n in [0, 1, 2]:
            return [n]
        return list(range(n, 0, -2))


class CumulativeProduct(RandomListTask):
    """
    Routine #8 from master list.

    Examples:

        (2 5 8 1 2) - (2 10 80 80 160)

    """
    name = 'cumulative_product'
    input_type = ListOfInts
    output_type = ListOfInts

    def func(self, x):
        last = 1
        output = []
        for n in x:
            last = n * last
            output.append(last)
        return output


class CumulativeSum(RandomListTask):
    """
    Routine #9 from master list.

    Examples:

        (2 5 8 1 2) - (2 7 15 16 18)

    """
    name = 'cumulative_sum'
    input_type = ListOfInts
    output_type = ListOfInts

    def func(self, x):
        last = 0
        output = []
        for n in x:
            last = n + last
            output.append(last)
        return output


class FlattenMapRepeatN(RandomListTask):
    """
    Routine #8 from master list.

    Examples:

        (3 1 6) - (3 3 3 1 6 6 6 6 6 6)

    """
    name = 'flatten_map_repeat_n_n_times'
    input_type = ListOfInts
    output_type = ListOfInts
    num_examples = 100  # fails under 20 examples

    def func(self, x):
        return [i for i in x for _ in range(i)]


def create_more_list_tasks():
    tasks = [
        RepeatN(),
        CountDown(),
        LastElement(),
        HeadthElement(),
        CountHead(),  # needs more examples!
        FlattenMapRange(),
        FlattenMapRangeReversed(),
        FlattenMapRangeSeries(),
        FlattenMapRangeHead(),
        Minus2Series(),
        CumulativeProduct(),
        CumulativeSum(),
        FlattenMapRepeatN(),

    ]
    names = []
    data = []
    for t in tasks:
        assert t.name not in names, f'Multiple tasks with the same name ({t.name}) exist!'
        names.append(t.name)
        data.append(t.json())

    with open(JSON_FILE, 'w') as f:
        json.dump(data, f)

    num_tasks = len(tasks)
    print(f'wrote {num_tasks} tasks to: {JSON_FILE}')


if __name__ == '__main__':
    create_more_list_tasks()
