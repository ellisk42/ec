#!/bin/bash

#bash run.sh S8_nojitter python bin/draw.py -t 720 --biasOptimal --contextual -R 1800 --trainset=S8_nojitter --doshaping
#bash run.sh S9_nojitter python bin/draw.py -t 720 --biasOptimal --contextual -R 1800 --trainset=S9_nojitter --doshaping
#bash run.sh S8long python bin/draw.py -t 5400 --biasOptimal --contextual -R 3600 --trainset=S8_nojitter --doshaping --testingTimeout 600 # stopped, since timed out
# bash run.sh S8long2 python bin/draw.py -t 3600 --biasOptimal --contextual -R 2200 --trainset=S8_nojitter --doshaping --testingTimeout 600 
#bash run.sh S9long python bin/draw.py -t 5400 --biasOptimal --contextual -R 3600 --trainset=S9_nojitter --doshaping --testingTimeout 600 #stopped, since would timeout, and S9_nojitter already learned ery well
# bash run.sh S9long2 python bin/draw.py -t 3600 --biasOptimal --contextual -R 2200 --trainset=S9_nojitter --doshaping --testingTimeout 600


bash run.sh S9.2 2000 20 python bin/draw.py -t 720 --biasOptimal --contextual -R 1800 --trainset=S9_nojitter --doshaping --testingTimeout 600
bash run.sh S8.2 2000 20 python bin/draw.py -t 3600 --biasOptimal --contextual -R 2400 --trainset=S8_nojitter --doshaping --testingTimeout 600
