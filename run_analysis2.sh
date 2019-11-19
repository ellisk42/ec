#!/bash/bin

bash run_analysis.sh S12.10.test4

# for all parse datsegs, get corresponding scores.
# also gets for randperms
python analysis/parsesGetPlannerScores.py


# for existing model-human dists, appends their planner scores
python analysis/modelParsesGetPlannerScores.py
