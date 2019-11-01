#!/bin/bash

CURRDIR=$(pwd)
# out=/tmp/analysis/outputs/"$DG_EXPT"_$(date +%Y-%m-%d_%H-%M-%S)
out="${CURRDIR}/analysis/outputs/analysis_$(date +%Y-%m-%d_%H-%M-%S)"
echo "Outputing to ${out}"
echo "Should be using conda envt: ecgood"

# 1) rules based analysis of behavior
DGDIR=/home/lucast4/drawgood/experiments

DG_EXPT=2.3
outthis=$out"_dgrulesmodel_"$DG_EXPT
echo "Assuming already extracted datall_${DG_EXPT}.pickle in drawgood"
echo "1) Rules-based modeling of behavior, running:"
cmd="python ${DGDIR}/modelRules.py $DG_EXPT > $outthis"
echo $cmd
# $cmd

DG_EXPT=2.3
outthis=$out"_dgrulesmodel_"$DG_EXPT
echo "Assuming already extracted datall_${DG_EXPT}.pickle in drawgood"
echo "1) Rules-based modeling of behavior, running:"
echo "python "$DGDIR"/modelRules.py $DG_EXPT > $outthis"
# python "$DGDIR"/modelRules.py $DG_EXPT > $outthis
