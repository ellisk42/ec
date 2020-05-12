#!/bin/bash

# ====== To make sure that exits if any python command fails uisntad of continuous, thus 
# drowining failures.
set -e # exits if any command fails
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command failed with exit code $?."' EXIT


# ====== "Neurips preparation, analysis from scratch (except starting from data that's already parsed and segmented)"

# 0) Preprocessing that I did not run anew, but instead took from previous analysis. Everything
# follwing this is new.
if false; then
    # 3) process model parses (--> datflat --> datseg)
    outthis=$out"_ecgetparses_"$EC_EXPT
    echo "3) parsing for ${EC_EXPT}:"
    # echo "python analysis/parse.py $EC_EXPT > $outthis"
    python analysis/parse.py $EC_EXPT
fi

if false; then
    outthis=$out"_ecgetdatflatseg_"$EC_EXPT
    echo "3) getting datflat/datseg for ${EC_EXPT}:"
    # echo "python analysis/parseProcess.py $EC_EXPT > $outthis"
    python analysis/parseProcess.py $EC_EXPT
fi

# 1) For all parse datsegs, get corresponding scores. also gets for randperms
if false; then
    python analysis/parsesGetPlannerScores.py 200118
fi

# 2) Get model-human distances. For all dreamcoder parses, compute distance to human
REMOVE_REDUNDANT_STROKES=1 # note these should be matched for getModelHumanDists and modelParsesGetPlannerScores

if false; then
    python analysis/getModelHumanDists.py S12.10.test5 0 $REMOVE_REDUNDANT_STROKES
    python analysis/getModelHumanDists.py S12.10.test5 1 $REMOVE_REDUNDANT_STROKES
    python analysis/getModelHumanDists.py S13.10.test5 0 $REMOVE_REDUNDANT_STROKES
    python analysis/getModelHumanDists.py S13.10.test5 1 $REMOVE_REDUNDANT_STROKES
fi

# 3) for existing model-human dists, appends their planner scores
python analysis/modelParsesGetPlannerScores.py test5 $REMOVE_REDUNDANT_STROKES


