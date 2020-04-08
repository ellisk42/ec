#!/bin/bash
module add openmind/singularity

bash run.sh practice 10000 20 python bin/sketch.py -i 10 -t 1800 --contextual -R 2400 --skiptesting --taskReranker unsolved
