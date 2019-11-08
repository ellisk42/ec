#!/bin/bash
# NOTE: "Should be using conda envt: ecgood"
# to plot summaries of DC experiments. 

CURRDIR=$(pwd)
# out=/tmp/analysis/outputs/"$DG_EXPT"_$(date +%Y-%m-%d_%H-%M-%S)
out="${CURRDIR}/analysis/outputs/analysis_$(date +%Y-%m-%d_%H-%M-%S)"
echo "Outputing to ${out}"
echo "Should be using conda envt: ecgood"

# 5) plot summaries to things
# EXPTLIST=(S9.2 S8.2.2)
# EXPTLIST=(S12.6.1 S12.6.2 S13.9 S12.8.1)
EXPTLIST=(S12.10 S13.10)
COMPARETOHUMAN=0
for EC_EXPT in ${EXPTLIST[@]}
do
    outthis=$out"_ecsummarize_"$EC_EXPT
    echo "5) summarizing ${EC_EXPT}:"
    cmd="python analysis/summarize.py $EC_EXPT $COMPARETOHUMAN"
    echo $cmd
    $cmd
done 

# === DONE
echo "DONE!"
