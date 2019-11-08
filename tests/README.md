Tests
=====

### Unit tests

To run unit tests from the root of the repo:
```
python -m unittest discover tests
```

Or, run the convenience script from the root of the repo:
```
./tests/run
```

### Integration tests


While refactoring, run the following tests.

Run the following 4 programs with small timeout to ensure no superficial bugs:

1. text.py
```bash
singularity exec container.img python bin/text.py -t 2 -RS 5 -i 2
```
2. list.py
```bash
singularity exec container.img python bin/list.py -t 2 -RS 5 -i 2
```
3. logo.py
```bash
singularity exec container.img python bin/logo.py -t 5 -RS 10 --biasOptimal -i 2
```
4. tower.py
```bash
singularity exec container.img python bin/tower.py -t 2 -RS 5 -i 2
```

Single test command:
```bash
singularity exec container.img python bin/text.py -t 2 -RS 5 -i 2 > text.out && singularity exec container.img python bin/list.py -t 2 -RS 5 -i 2 > list.out && singularity exec container.img python bin/logo.py -t 5 -RS 10 --biasOptimal -i 2 > logo.out
```

More extensive test command:
```bash
singularity exec container.img python bin/tower.py -t 600 --pseudoCounts 30 \
            --tasks new --aic 1.0 --structurePenalty 1.5 --topK 2 --arity 3 \
            --maximumFrontier 5 -i 10 --storeTaskMetrics --split 0.5 \
            --testingTimeout 600 --biasOptimal --contextual --primitives new --recognitionTimeout 3600 -RS 5000
```

Also run some graph.py tests from the root of the repo:
```bash
bash tests/integration/runtests.sh
``` 
 