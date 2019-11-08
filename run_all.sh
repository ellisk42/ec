#!/bash/bin

# script indicating analysis sequeence

# 0) THIS CONTAINS EVERYTHING?
bash analysis.sh

# OTHERWISE, my attempt at piecing things together.
# 1) get summary figuures
bash run_summarize.sh

# 2) get parses
python analysis/parse.py "S8.2.2"

# 3) do analyses
python analyses/preprocess.py S8.2.2