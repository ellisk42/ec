#!/bash/bin
module add openmind/singularity

#bash run_nogpu.sh S9.2_parse 10000 1 python analysis/parse.py "S9.2"
# bash run_nogpu.sh S8.2.2_parse 15000 1 python analysis/parse.py "S8.2.2"


bash run_nogpu.sh S12.10.test5_parse 15000 2 python analysis/parse.py S12.10.test5
bash run_nogpu.sh S13.10.test5_parse 15000 2 python analysis/parse.py S13.10.test5
