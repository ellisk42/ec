# Wave 3 Request ([Task](#task) &middot; [Deliverables](#deliverables))

## Task
Run your model on 100 problems for 5 runs of 5 orderings of 11 trials for 10min per trial in an online setting:

- **100 problems** The [100 problems](./problems.md) use two DSLs:
  - Problems 1–80 will use [a DSL](./dsl.md) with just the numbers 0..9.
  - Problems 81–100 use [the same DSL](./dsl.md) extended to include 0..99.
- **5 runs** For each combination of the other factors, run the algorithm 5 times. This will help us assess variance in the algorithm due to randomization.
- **5 orderings** I have provided 5 orderings of the trials for each problem. Please run each set as given, i.e. without shuffling or reordering. This will help us assess variance across trial ordering and ensure all models are learning from comparable data.
- **11 trials** Each trial set contains 11 examples. For trial N+1, run the model with examples 1 to N available as training data. Hold out example N+1 as your test datum. This gives us the ability to plot learning curves similar to the human experiment. For trial 1, the training set will be empty.
- **10min timeout** Each trial will run for a total of 10min. For a particular problem, trial order, and run, search for 10min on Trial 1, then another 10min on Trial 2, and so on, for a total of 110min of search.
- **Online setting** We’ll be comparing model performance to humans in an [online learning](https://en.wikipedia.org/wiki/Online_machine_learning) task. Conceptually, think of each run through a series of trials as a single agent sequentially predicting the output for 11 different inputs, observing the correct output, updating their fitness/scoring/evaluation function, and searching for a better hypothesis. Practically, please setup your model so that for a particular problem, trial order, and run, Trial N+1 starts where Trial N finished, reusing the computation from Trials 1 to N to hotstart Trial N+1.

JSON files containing trials for each concept can be found in [`json/`](./json). They are named according to problem and ordering, e.g. `c007_4.json` contains trials for problem 7, ordering 4. Each file contains the problem `id`, a string describing the `program` in a much richer DSL, and the `data`. Each datum is an i/o pair containing an `i`nput, and an `o`utput. You can convert from JSON to CSV or TXT with [`json2.py`](../../src/json2.py) if those formats are easier for your model to use.


<!-- If any of these conditions don’t make sense for your model (e.g. the algorithm is deterministic and makes no random choices), reach out to me, and we can discuss appropriate modifications. Please reach out with any questions you might have. -->

## Deliverables

Please provide 4 files:

1. `predictions.csv`: a CSV describing the program used to make the **final predictions** for the test example at the end of each trial. These predictions will be used to assess the ultimate performance of the model.

2. `bests.csv`: a CSV describing all the **best-so-far programs** considered during each trial. A best-so-far program is one which scores better than all programs previously considered during search. For example, if program 193 scores better than the 192 previously considered programs, it is a best-so-far program. Program 1 of each trial is always a best-so-far program. These will be used to assess how sensitive each model is to additional search.

3. `samples.csv`: a CSV describing **10,000 programs uniformly sampled** from all programs considered during all runs. You can [do this easily with reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling#With_random_sort). These will be used to analyze how models move through search space.

4. `description.md`: a 2-paragraph description of the model:
   - 1 paragraph describing how the model works (i.e. basic mechanics, principles of operation, relevant citations)
   - 1 paragraph giving simulation-specific details (i.e. parameters, offline training, background knowledge, metarules, etc.)

You are welcome to send any additional files that you think helpful (e.g. model output, lists of metarules).

### CSV Format

The CSV files should have these columns:
- `problem` (quoted string): the problem id, use the 'c???' labels
- `run` (int): the run number, 1-indexed
- `order` (int): the trial order, 1-indexed
- `trial` (int): the trial number, 1-indexed
- `program` (quoted string): a string representation of the program in the DSL
- `cpu` (float ): the total CPU time that passed before finding this program (summed across cores if using multiple CPUs)
- `count` (int): the total number of programs considered before this program

Here’s an example header and entry from such a file:
```csv
problem,run,order,trial,program,cpu,count
”c003”,3,5,2,”(lambda $0)”,0.00034,13
```

### Program Format

I will be recomputing program outputs and other properties of the programs (e.g. length, prior, likelihood) post hoc, so it's critical that you report your programs using the [canonical DSL](./dsl.md). You may need to write a converter if your model uses an alternative representation (e.g. converting explicit recursion to `fix` or predicate invention to `lambda`). To check the accuracy of any conversions, an interpreter for the DSL is available by installing [`ec`](https://github.com/joshrule/ec) and using [`bin/list-routines.py`](https://github.com/joshrule/ec/blob/master/bin/list-routines.py). For example:

```bash
# Note the escaped "\$" to avoid bash string interpolation.
$ python bin/list-routines.py model-comparison-9 "(lambda (cons (head \$0) empty))" "[1,2,3,4]"
[1]
```
