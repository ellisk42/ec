#!/bin/bash
# NOTE: "Should be using conda envt: ecgood"

# EC_EXPT=S8.2.2
# EC_EXPT=S9.2
EC_EXPT=$1
echo $EC_EXPT
DOPARSE=false # false, skip parsing (i.e, alreadyd one), true, then do
DOSEG=false

CURRDIR=$(pwd)
# out=/tmp/analysis/outputs/"$DG_EXPT"_$(date +%Y-%m-%d_%H-%M-%S)
out="${CURRDIR}/analysis/outputs/analysis_$(date +%Y-%m-%d_%H-%M-%S)"
echo "Outputing to ${out}"
echo "Should be using conda envt: ecgood"

##################################33
# 1) rules based analysis of behavior

if false; then
    DGDIR=/home/lucast4/drawgood/experiments
    DG_EXPT=2.3
    outthis=$out"_dgrulesmodel_"$DG_EXPT
    echo "Assuming already extracted datall_${DG_EXPT}.pickle in drawgood"
    echo "1) Rules-based modeling of behavior, running:"
    echo "python "$DGDIR"/modelRules.py $DG_EXPT > $outthis"
    python "$DGDIR"/modelRules.py $DG_EXPT > $outthis

    DG_EXPT=2.2
    outthis=$out"_dgrulesmodel_"$DG_EXPT
    echo "Assuming already extracted datall_${DG_EXPT}.pickle in drawgood"
    echo "1) Rules-based modeling of behavior, running:"
    echo "python "$DGDIR"/modelRules.py $DG_EXPT > $outthis"
    python "$DGDIR"/modelRules.py $DG_EXPT > $outthis
fi

# 2) get parses
if false; then
    # NOTE: have to enter the correct task names in makeDrawTasks before can run.
    # NOTE, copy S8.2./2 below
    EC_EXPT=S8fixedprim
    outthis=$out"_ecgetparses_"$EC_EXPT
    echo "2) getting parses for ${EC_EXPT}:"
    echo "python analysis/parse.py $EC_EXPT > $outthis"
    python analysis/parse.py $EC_EXPT > $outthis

    EC_EXPT=S9fixedprim
    outthis=$out"_ecgetparses_"$EC_EXPT
    echo "2) getting parses for ${EC_EXPT}:"
    echo "python analysis/parse.py $EC_EXPT > $outthis"
    python analysis/parse.py $EC_EXPT > $outthis

    outthis=$out"_ecgetparses_"$EC_EXPT
    echo "2) getting parses for ${EC_EXPT}:"
    echo "python analysis/parse.py $EC_EXPT > $outthis"
    python analysis/parse.py $EC_EXPT > $outthis
fi

if $DOPARSE; then
    # 3) process model parses (--> datflat --> datseg)
    outthis=$out"_ecgetparses_"$EC_EXPT
    echo "3) parsing for ${EC_EXPT}:"
    # echo "python analysis/parse.py $EC_EXPT > $outthis"
    python analysis/parse.py $EC_EXPT
fi

if $DOSEG; then
    outthis=$out"_ecgetdatflatseg_"$EC_EXPT
    echo "3) getting datflat/datseg for ${EC_EXPT}:"
    # echo "python analysis/parseProcess.py $EC_EXPT > $outthis"
    python analysis/parseProcess.py $EC_EXPT
fi

# 4) get human-model distances
outthis=$out"_ecmodelhumandists_"$EC_EXPT
echo "4) getting modelHumanDists for ${EC_EXPT}:"
echo "python analysis/getModelHumanDists.py $EC_EXPT > $outthis"
python analysis/getModelHumanDists.py $EC_EXPT

# 5) plot summaries to things
outthis=$out"_ecsummarize_"$EC_EXPT
echo "5) summarizing ${EC_EXPT}:"
echo "python analysis/summarize.py $EC_EXPT > $outthis"
python analysis/summarize.py $EC_EXPT

# === DONE
echo "DONE!"