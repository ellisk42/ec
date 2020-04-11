#!/bin/bash
module add openmind/singularity

#bash run.sh practice 10000 20 python bin/sketch.py -i 10 -t 1800 --contextual -R 2400 --skiptesting --taskReranker unsolved
#bash run.sh practice 10000 20 python bin/sketch.py -i 10 -t 1800 --contextual -R 2400 --skiptesting

#bash run.sh practice 10000 20 python bin/sketch.py -i 10 -t 720 --contextual -R 720 --skiptesting --taskReranker unsolved
bash run.sh practicenoCNN 10000 20 python bin/sketch.py -i 10 -t 720 --contextual -R 720 --skiptesting --taskReranker unsolved

