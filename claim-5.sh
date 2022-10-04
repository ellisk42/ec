

mkdir -p artifact_out

# Row 1: base
for i in 0 1 2 3; do
  echo "Running base iteration $i"
  pypy3 bin/sasquatch.py -v -a 0  -i -t --iteration $i &> artifact_out/base-$i.stderr
done

# Row 2: +Bayes
for i in 0 1 2 3; do
  echo "Running +Bayes iteration $i"
  pypy3 bin/sasquatch.py -v -a 0  -i -t --iteration $i -r &> artifact_out/bayes-$i.stderr
done

# Row 1: +VS
for i in 0 1 2 3; do
  echo "Running +VS iteration $i"
  pypy3 bin/sasquatch.py -v -a 1  -i -t --iteration $i &> artifact_out/vs-$i.stderr
done

# Row 1: +Bayes +VS
for i in 0 1 2 3; do
  echo "Running +Bayes+VS iteration $i"
  pypy3 bin/sasquatch.py -v -a 1  -i -t --iteration $i -r &> artifact_out/bayes_vs-$i.stderr
done

