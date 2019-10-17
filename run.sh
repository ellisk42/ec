#!/bin/bash

fn=jobs/"$1"_$(date +%Y-%m-%d_%H-%M-%S)
echo "Executing:"
echo "${@:2}"
echo "Exporting to $fn"
srun --job-name="$1" --output=jobs/"$1"_$(date +%Y-%m-%d_%H-%M-%S) --ntasks=1 --mem-per-cpu=1500 --cpus-per-task 15 -p tenenbaum --gres=gpu --time=48:00:00 singularity exec --nv container.img ${@:2} \
    &
