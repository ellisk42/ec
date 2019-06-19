Creating New Domains
====================

To create new domains of problems to solve, a number of things must be done. Follow the steps below to get started.

# Table of contents
1. [Select a Domain](#select-a-domain)
2. [Create a Domain Script](#create-a-domain-script)
    1. [Add Python Imports](#add-python-imports)
    2. [Create a commandline for the script](#create-a-commandline-for-the-script)
    3. [Define primitives for the domain](#define-primitives-for-the-domain)
    4. [Update OCaml Primitives](#update-ocaml-primitives)
    5. [Create Training and Testing Tasks](#create-training-and-testing-tasks)
    5. [Create an `ecIterator`](#create-an-eciterator)
    5. [Final Script](#final-script)
3. [Run the Script](#run-the-script)

# Select a Domain

Select an appropriate domain to train the DreamCoder algorithm to solve tasks within.

In our example below, the domain will be adding positive integers, with the tasks involving addition of different numbers. It's a trivial, toy example to demonstrate the different components of the codebase that you need to get started.

# Create a Domain Script

First we need to create a script that will have the primitives we care about, and the tasks we want our program to solve.

Create a Python file in the `bin/` directory that will be our script. In this example, we'll call ours `bin/incr.py`.

We can model our script after one of the other scripts in `bin/`, e.g. `bin/list.py` or `bin/text.py`.

### Add Python Imports

Lets start by adding some imports to our script:
```python
import datetime
import os
import random

import binutil

from dreamcoder.ec import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint
from dreamcoder.utilities import numberOfCPUs
```
The `import binutil` is a funky workaround for the directory structure of this repo. You can safely ignore it for now.

The other imports are going to give us the building blocks for defining a commandline, creating primitives, and other essential components to solving tasks within our new domain.

### Create a commandline for the script

To create a Python `argparse` commandline for our script, we can import the `commandlineArguments()` function from the `ec.py` module, which has most of the parameters that the DreamCoder algorithm needs in order to run successfully.

Here's an example with some defaults set:
```python
args = commandlineArguments(
    enumerationTimeout=10, activation='tanh',
    iterations=10, recognitionTimeout=3600,
    a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
    helmholtzRatio=0.5, structurePenalty=1.,
    CPUs=numberOfCPUs())
```

Next, lets begin to define our domain.

### Define primitives for the domain

We will create a list of primitives for our toy example next.

Each member of our list must be an instance of the `Primitive` class where each primitive has a unique name that binds it to its corresponding OCaml code (discussed later), a type imported from `dreamcoder/type.py`, and a lambda function: `Primitive(name, type, func)`.
```python
def _incr(x): return lambda x: x + 1
def _incr2(x): return lambda x: x + 2

primitives = [
    Primitive("incr", arrow(tint, tint), _incr),
    Primitive("incr2", arrow(tint, tint), _incr2),
]
```

Then create a Grammar from the primitives:
```python
grammar = Grammar.uniform(primitives)
```

Note that *new primitives cannot currently be created without modifying OCaml code*! See the next section to continue.

### Update OCaml Primitives

A limitation of the current architecture is that the primitives must also be defined in both the Python frontend and the OCaml backend (in `solvers/program.ml`).

If we open up that file, we will see that there is no primitive for `incr2`, which would raise a runtime error for our new domain. So edit the `solvers/program.ml` file to add an "incr2" primitive:
```diff
 let primitive_increment = primitive "incr" (tint @> tint) (fun x -> 1+x);;
+let primitive_increment2 = primitive "incr2" (tint @> tint) (fun x -> 2+x);;
```

This will also mean we need to rebuild the OCaml binaries:
```
make clean
make
```
See the README at the root of the repo (specifically the "Building the OCaml binaries" section) for more info.

### Create Training and Testing Tasks

Now that we have defined our primitives and grammar, we can create some training and testing tasks in our domain.

First off, lets define a helper function that will add some number `N` to a pseudo-random number:
```python
def addN(n):
    x = random.choice(range(500))
    return {"i": x, "o": x + n}
```
The return value is a dictionary format we will use to store the inputs and the outputs for each task.

Each task will consist of 3 things:
1. a name
2. a mapping from input to output type (e.g. `arrow(tint, tint)`)
3. a list of input-output pairs

The input-output pairs should be a list of tuples (input, output) where each input is itself a tuple.

Lets define a helper function to do the work of creating tasks for us:
```python
def get_tint_task(item):
    return Task(
        item["name"],
        arrow(tint, tint),
        [((ex["i"],), ex["o"]) for ex in item["examples"]],
    )
```

After defining our helper functions, we can add some training data:
```python
# Training data
def add1(): return addN(1)
def add2(): return addN(2)
def add3(): return addN(3)
training_examples = [
    {"name": "add1", "examples": [add1() for _ in range(5000)]},
    {"name": "add2", "examples": [add2() for _ in range(5000)]},
    {"name": "add3", "examples": [add3() for _ in range(5000)]},
]
training = [get_tint_task(item) for item in training_examples]
```

Following that, lets add a smaller amount of testing data:
```python
# Testing data
def add4(): return addN(4)
testing_examples = [
    {"name": "add4", "examples": [add4() for _ in range(500)]},
]
testing = [get_tint_task(item) for item in testing_examples]
```

### Create an `ecIterator`

Finally, to get the DreamCoder algorithm to run over our tasks, we need to create an `ecIterator` in our script as follows:
```python
generator = ecIterator(grammar,
                       training,
                       testingTasks=testing,
                       **args)
for i, _ in enumerate(generator):
    print('ecIterator count {}'.format(i))
```

This will iterate over the wake and sleep cycles for our tasks.

### Final script

The end result of our domain script might look something like this:
```python
import datetime
import os
import random

import binutil  # required to import from dreamcoder modules

from dreamcoder.ec import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint
from dreamcoder.utilities import numberOfCPUs

# Primitives
def _incr(x): return lambda x: x + 1
def _incr2(x): return lambda x: x + 2


def addN(n):
    x = random.choice(range(500))
    return {"i": x, "o": x + n}


def get_tint_task(item):
    return Task(
        item["name"],
        arrow(tint, tint),
        [((ex["i"],), ex["o"]) for ex in item["examples"]],
    )


if __name__ == "__main__":

    # Options more or less copied from list.py

    args = commandlineArguments(
        enumerationTimeout=10, activation='tanh',
        iterations=10, recognitionTimeout=3600,
        a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
        helmholtzRatio=0.5, structurePenalty=1.,
        CPUs=numberOfCPUs())

    timestamp = datetime.datetime.now().isoformat()
    outdir = 'experimentOutputs/demo/'
    os.makedirs(outdir, exist_ok=True)
    outprefix = outdir + timestamp
    args.update({"outputPrefix": outprefix})

    # Create list of primitives

    primitives = [
        Primitive("incr", arrow(tint, tint), _incr),
        Primitive("incr2", arrow(tint, tint), _incr2),
    ]

    # Create grammar

    grammar = Grammar.uniform(primitives)

    def add1(): return addN(1)
    def add2(): return addN(2)
    def add3(): return addN(3)

    # Training data

    training_examples = [
        {"name": "add1", "examples": [add1() for _ in range(5000)]},
        {"name": "add2", "examples": [add2() for _ in range(5000)]},
        {"name": "add3", "examples": [add3() for _ in range(5000)]},
    ]
    training = [get_tint_task(item) for item in training_examples]

    # Testing data

    def add4(): return addN(4)

    testing_examples = [
        {"name": "add4", "examples": [add4() for _ in range(500)]},
    ]
    testing = [get_tint_task(item) for item in testing_examples]

    # EC iterate

    generator = ecIterator(grammar,
                           training,
                           testingTasks=testing,
                           **args)
    for i, _ in enumerate(generator):
        print('ecIterator count {}'.format(i))
```

Reminder: Don't forget to rebuild the OCaml binaries before proceeding to the next step!

# Run the Script

To run the script, run as follows:
```
python bin/incr.py -t 2 --testingTimeout 2
```

So within a singularity container for 2 iterations (`-i 2`):
```
singularity exec container.img python bin/incr.py -t 2 --testingTimeout 2 -i 2
```

Our script is a trivial example so we should not expect to see much improvement over the course of the iterations. The program should be solved during the first iteration.

The training tasks are straightforward, so we should expect to see something like the console output showing that the algorithm found a solution for each task:
```
Generative model enumeration results:
HIT add1 w/ (lambda (incr $0)) ; log prior = -2.197225 ; log likelihood = 0.000000
HIT add2 w/ (lambda (incr2 $0)) ; log prior = -2.197225 ; log likelihood = 0.000000
HIT add3 w/ (lambda (incr (incr2 $0))) ; log prior = -3.295837 ; log likelihood = 0.000000
```

The console output should show that at some point the algorithm solved the testing task as well:
```
HIT add4 w/ (lambda (incr2 (incr2 $0))) ; log prior = -3.295837 ; log likelihood = 0.000000
```

For more complicated examples, where the tasks are not all immediately solved in the first iteration, loss should drop with each iteration as the algorithm improves.

For more information about running scripts, in the README.md, see "Running tasks from the commandline". Also read about graphing the results when testing more complicated domains.
