#!/bin/bash

fn=jobs/"$1"_$(date +%Y-%m-%d_%H-%M-%S)
echo "Executing:"
echo "${@:4}"
echo "Exporting to $fn"
echo "Running command:"
echo "srun --job-name="$1" --output=jobs/"$1"_$(date +%Y-%m-%d_%H-%M-%S) --ntasks=1 --mem-per-cpu=$2 --cpus-per-task $3 -p tenenbaum --gres=gpu --time=48:00:00 singularity exec --nv container.img ${@:4} \
    &"
srun --job-name="$1" --output=jobs_nogpu/"$1"_$(date +%Y-%m-%d_%H-%M-%S) --ntasks=1 --mem-per-cpu=$2 --cpus-per-task $3 -p tenenbaum --time=48:00:00 singularity exec --nv container.img ${@:4} \
 &
    
