#!/bin/bash
############ compilation of plots to summarize any given dreamcoder trained model.
# This would be the first thing you run after getting a trained dreamcoder model.

# ====== To make sure that exits if any python command fails uisntad of continuous, thus 
# drowining failures.
set -e # exits if any command fails
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command failed with exit code $?."' EXIT

# 1) Summary plots
# EC_EXPT="S12.inv.S12skew.1.test"
# python analysis/summarize.py S12.inv.S12skew.1.test 0

# EC_EXPT="S12.inv.S12grate.1.test"
# python analysis/summarize.py S12.inv.S12grate.1.test 0

# EC_EXPT="S13.inv.S13grate.1.test" 
# python analysis/summarize.py S13.inv.S13grate.1.test 0

# EC_EXPT="S12.inv.base.test"
# python analysis/summarize.py S12.inv.base.test 0

# EC_EXPT="S13.inv.base.test"
# python analysis/summarize.py S13.inv.base.test 0

# 2) Parse
# EC_EXPT="S12.inv.S12skew.1.test"
# python analysis/parse.py S13.10.test5
EC_EXPT="S12.inv.S12skew.1.test"
python analysis/parse.py S12.inv.base.test
