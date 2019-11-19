#!/bash/bin

# script indicating analysis sequeence
EC1=S12.10.test5
EC2=S13.10.test5
plannerscorever=test5
out="${CURRDIR}/analysis/outputs/analysis_$(date +%Y-%m-%d_%H-%M-%S)"

################# MAIN ANALYSES AND SUMMARIES
outthis=$out"_analysis_"$EC1
cmd="bash run_analysis.sh $EC1"
echo $cmd
$cmd

outthis=$out"_analysis_"$EC2
cmd="bash run_analysis.sh $EC2"
echo $cmd
$cmd

############## PLANNER REQEIGHTING OF DREAMCODER SCORES
# 5) next to datseg, extract planner scores for the exact same parses
outthis=$out"_ecparsegetplannerscore"
cmd="python analysis/parsesGetPlannerScores.py > $outthis"
echo $cmd
$cmd

# 6) combines steps 4 and 5 into one combined summary dict.
outthis=$out"_ecmodelparsecombine"
cmd="python analysis/modelParsesGetPlannerScores.py $plannerscorever > $outthis"
echo $cmd
$cmd

# OTHERWISE, my attempt at piecing things together.
# 1) get summary figuures
# bash run_summarize.sh

# 2) get parses
# python analysis/parse.py "S8.2.2"

# 3) do analyses
# python analyses/preprocess.py S8.2.2