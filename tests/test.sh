#!/bin/bash

# usage (from repo root):
#
#     nohup bash ./tests/test.sh & echo $! > pidfile
#

singularity exec container.img python bin/text.py -t 2 -RS 5 -i 2 > text.out &&\
    singularity exec container.img python bin/list.py -t 2 -RS 5 -i 2 > list.out && \
    singularity exec container.img python bin/logo.py -t 5 -RS 10 --biasOptimal -i 2 > logo.out && \
    singularity exec container.img python bin/regexes.py -i 1 -t 1 -RS 10 -R 10 \
        --primitives reduced --tasks new --maxTasks 256 --ll_cutoff bigram --split 0.5 --pseudoCounts 30 \
        -l -1000000 --aic -1000000 --structurePenalty 1.5 --topK 2 --arity 3 --primitives strConst \
        --use_str_const -g > regexes.out && \
    singularity exec container.img python bin/tower.py -t 2 -RS 5 -i 2 -l -1000000 --aic -1000000 \
        --tasks new --primitives new -g > tower.out && \
    singularity exec container.img python bin/scientificLaws.py -i 1 -t 1 -RS 10 -R 10 \
        --pseudoCounts 30 -l -1000000 --aic -1000000 -g > scientificLaws.out && \
    singularity exec container.img python bin/rational.py -i 1 -t 1 --testingTimeout 1 \
        -RS 10 -R 10 --pseudoCounts 30 -l -1000000 --aic -1000000 -g > rational.out

echo "exited with $?" > nohup.exitcode
