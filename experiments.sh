#!/bin/bash

#bash run.sh S8_nojitter python bin/draw.py -t 720 --biasOptimal --contextual -R 1800 --trainset=S8_nojitter --doshaping
#bash run.sh S9_nojitter python bin/draw.py -t 720 --biasOptimal --contextual -R 1800 --trainset=S9_nojitter --doshaping
#bash run.sh S8long python bin/draw.py -t 5400 --biasOptimal --contextual -R 3600 --trainset=S8_nojitter --doshaping --testingTimeout 600 # stopped, since timed out
# bash run.sh S8long2 python bin/draw.py -t 3600 --biasOptimal --contextual -R 2200 --trainset=S8_nojitter --doshaping --testingTimeout 600 
#bash run.sh S9long python bin/draw.py -t 5400 --biasOptimal --contextual -R 3600 --trainset=S9_nojitter --doshaping --testingTimeout 600 #stopped, since would timeout, and S9_nojitter already learned ery well
# bash run.sh S9long2 python bin/draw.py -t 3600 --biasOptimal --contextual -R 2200 --trainset=S9_nojitter --doshaping --testingTimeout 600


#bash run.sh S9.2 10000 20 python bin/draw.py -t 720 --biasOptimal --contextual -R 1800 --trainset=S9_nojitter --doshaping --testingTimeout 600
#bash run.sh S8.2 10000 20 python bin/draw.py -t 3600 --biasOptimal --contextual -R 2400 --trainset=S8_nojitter --doshaping --testingTimeout 600
#bash run.sh S8.2.1 10000 20 python bin/draw.py -t 720 --biasOptimal --contextual -R 1800 --trainset=S8_nojitter --doshaping --testingTimeout 600
# bash run.sh S8.2.2 20000 20 python bin/draw.py -t 600 --biasOptimal --contextual -R 1200 --trainset=S8_nojitter --doshaping --testingTimeout 600

#bash run.sh S10 10000 20 python bin/draw.py -i 10 -t 720 --biasOptimal --contextual -R 1800 --trainset=S10
#bash run.sh S10shaping 10000 20 python bin/draw.py -i 10 -t 720 --biasOptimal --contextual -R 1800 --trainset=S10 --doshaping
#bash run.sh ignore 10000 20 python bin/draw.py -i 10 -t 1440 --biasOptimal --contextual -R 1800 --trainset=S10 --doshaping

# bash run.sh S10good 8000 20 python bin/draw.py -i 10 -t 720 --biasOptimal --contextual -R 1800 --trainset=S10
#bash run.sh S10goodshaping 5000 20 python bin/draw.py -i 10 -t 720 --biasOptimal --contextual -R 1800 --trainset=S10 --doshaping
#bash run.sh S10goodlong 10000 20 python bin/draw.py -i 6 -t 2000 --biasOptimal --contextual -R 2400 --trainset=S10

bash run.sh S12 10000 20 python bin/draw.py -i 6 -t 720 --biasOptimal --contextual -R 1800 --trainset=S12 --testingTimeout 600 --dopruning
bash run.sh S13 10000 20 python bin/draw.py -i 6 -t 720 --biasOptimal --contextual -R 1800 --trainset=S13 --testingTimeout 600 --dopruning


