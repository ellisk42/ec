#!/bash/bin

python analysis/getModelHumanDists.py S13.10.test4 0
python analysis/getModelHumanDists.py S13.10.test4 1
python analysis/parsesGetPlannerScores.py
python analysis/modelParsesGetPlannerScores.py 13