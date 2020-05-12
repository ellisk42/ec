#!/bin/bash
module add openmind/singularity

#bash run.sh practice 10000 20 python bin/sketch.py -i 10 -t 1800 --contextual -R 2400 --skiptesting --taskReranker unsolved
#bash run.sh practice 10000 20 python bin/sketch.py -i 10 -t 1800 --contextual -R 2400 --skiptesting

#bash run.sh practice2 5000 15 python bin/sketch.py -i 10 -t 1800 --contextual -R 1800 --skiptesting --taskReranker unsolved
#bash run.sh practice2More 10000 25 python bin/sketch.py -i 10 -t 1800 --contextual -R 1800 --skiptesting --taskReranker unsolved
#bash run.sh practicenoCNN 5000 15 python bin/sketch.py -i 10 -t 1800 --no-recognition --skiptesting --taskReranker unsolved

#bash run.sh sketchv2.1 10000 20 python bin/sketch.py -i 10 -t 1800 --contextual -RS 10000 --trainset="v2.1" --skiptesting
#bash run.sh sketchv2.1lo 5000 10 python bin/sketch.py -i 10 -t 1800 --contextual -RS 10000 --trainset="v2.1" --skiptesting

bash run.sh sketchv2.2 10000 20 python bin/sketch.py -i 10 -t 1800 --contextual -RS 10000 --trainset=v2.1 --skiptesting



