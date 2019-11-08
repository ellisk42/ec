#!/bin/bash
module add openmind/singularity

#bash run.sh S8_nojitter python bin/draw.py -t 720 --biasOptimal --contextual -R 1800 --trainset=S8_nojitter --doshaping
#bash run.sh S9_nojitter python bin/draw.py -t 720 --biasOptimal --contextual -R 1800 --trainset=S9_nojitter --doshaping
#bash run.sh S8long python bin/draw.py -t 5400 --biasOptimal --contextual -R 3600 --trainset=S8_nojitter --doshaping --testingTimeout 600 # stopped, since timed out
# bash run.sh S8long2 python bin/draw.py -t 3600 --biasOptimal --contextual -R 2200 --trainset=S8_nojitter --doshaping --testingTimeout 600 
#bash run.sh S9long python bin/draw.py -t 5400 --biasOptimal --contextual -R 3600 --trainset=S9_nojitter --doshaping --testingTimeout 600 #stopped, since would timeout, and S9_nojitter already learned ery well
# bash run.sh S9long2 python bin/draw.py -t 3600 --biasOptimal --contextual -R 2200 --trainset=S9_nojitter --doshaping --testingTimeout 600


#bash run.sh S9.2 10000 20 python bin/draw.py -t 720 --biasOptimal --contextual -R 1800 --trainset=S9_nojitter --doshaping --testingTimeout 600
#bash run.sh S8.2 10000 20 python bin/draw.py -t 3600 --biasOptimal --contextual -R 2400 --trainset=S8_nojitter --doshaping --testingTimeout 600
#bash run.sh S8.2.1 10000 20 python bin/draw.py -t 720 --biasOptimal --contextual -R 1800 --trainset=S8_nojitter --doshaping --testingTimeout 600
# bash run.sh S8.2.2 20000 20 python bin/draw.py -t 600 --biasOptimal --contextual -R 1200 --trainset=S8_nojitter --doshaping --testingTimeout 600

#bash run.sh S10 10000 20 python bin/draw.py -i 10 -t 720 --biasOptimal --contextual -R 1800 --trainset=S10
#bash run.sh S10shaping 10000 20 python bin/draw.py -i 10 -t 720 --biasOptimal --contextual -R 1800 --trainset=S10 --doshaping
#bash run.sh ignore 10000 20 python bin/draw.py -i 10 -t 1440 --biasOptimal --contextual -R 1800 --trainset=S10 --doshaping

# bash run.sh S10good 8000 20 python bin/draw.py -i 10 -t 720 --biasOptimal --contextual -R 1800 --trainset=S10
#bash run.sh S10goodshaping 5000 20 python bin/draw.py -i 10 -t 720 --biasOptimal --contextual -R 1800 --trainset=S10 --doshaping
#bash run.sh S10goodlong 10000 20 python bin/draw.py -i 6 -t 2000 --biasOptimal --contextual -R 2400 --trainset=S10

#bash run.sh S12 10000 20 python bin/draw.py -i 6 -t 720 --biasOptimal --contextual -R 1800 --trainset=S12 --testingTimeout 600 --dopruning
#bash run.sh S13 10000 20 python bin/draw.py -i 6 -t 720 --biasOptimal --contextual -R 1800 --trainset=S13 --testingTimeout 600 --dopruning

#bash run.sh S12notest 15000 20 python bin/draw.py -i 6 -t 600 --biasOptimal --contextual -R 1200 --trainset=S12 --dopruning --skiptesting
#bash run.sh S13notest 15000 20 python bin/draw.py -i 6 -t 600 --biasOptimal --contextual -R 1200 --trainset=S13 --dopruning --skiptesting

#bash run.sh S12.1 5000 20 python bin/draw.py -i 40 -t 600 --biasOptimal --contextual -R 1200 --trainset=S12 --dopruning --skiptesting --taskReranker default --taskBatchSize 8 -a 2
#bash run.sh S13.1 5000 20 python bin/draw.py -i 40 -t 600 --biasOptimal --contextual -R 1200 --trainset=S13 --dopruning --skiptesting --taskReranker default --taskBatchSize 8 -a 2
#bash run.sh S9.2stagingnoshape 5000 20 python bin/draw.py -i 40 -t 720 --biasOptimal --contextual -R 800 --trainset=S9_nojitter --skiptesting --taskReranker default --taskBatchSize 8 -a 2

#bash run.sh S12.2 14000 20 python bin/draw.py -i 10 -t 1000 --biasOptimal --contextual -R 1800 --trainset=S12 --dopruning --skiptesting -a 2
#bash run.sh S13.2 14000 20 python bin/draw.py -i 10 -t 1000 --biasOptimal --contextual -R 1800 --trainset=S13 --dopruning --skiptesting -a 2

#bash run.sh S12.3 15000 20 python bin/draw.py -i 10 -t 2400 --biasOptimal --contextual -R 1800 --trainset=S12 --dopruning --skiptesting -a 2
#bash run.sh S12.3fullprim 15000 20 python bin/draw.py -i 10 -t 1000 --biasOptimal --contextual -R 1800 --trainset=S12 --skiptesting -a 2
#bash run.sh S12.3shape 10000 20 python bin/draw.py -i 10 -t 1000 --biasOptimal --contextual -R 1800 --trainset=S12 --skiptesting -a 2 --doshaping
#bash run.sh S9.2replicate 10000 20 python bin/draw.py -t 720 --biasOptimal --contextual -R 1800 --trainset=S9_nojitter --doshaping --skiptesting -a 2

#bash run.sh S12.4 5000 20 python bin/draw.py -i 10 -t 10 --biasOptimal --contextual -R 10 --trainset=S12 --dopruning --skiptesting -a 2 --taskReranker split --taskSplit 11,11,20

#bash run.sh S12.3.1 10000 20 python bin/draw.py -i 10 -t 1440 --biasOptimal --contextual -R 1800 --trainset=S12 --dopruning --skiptesting -a 2
#bash run.sh S12.5.1 10000 20 python bin/draw.py -i 20 -t 450 --biasOptimal --contextual -R 450 --trainset=S12 --dopruning --skiptesting -a 2 --taskReranker split --taskSplit 10,17,30
#bash run.sh S12.5.2 10000 20 python bin/draw.py -i 20 -t 450 --biasOptimal --contextual -R 450 --trainset=S12 --skiptesting -a 2 --taskReranker split --taskSplit 10,17,30 

#bash run.sh S12.6.1 12000 20 python bin/draw.py -i 20 -t 1800 --biasOptimal --contextual -R 1800 --trainset=S12 --dopruning --skiptesting -a 2 --taskReranker split --taskSplit 0,11,31
#bash run.sh S12.6.2 20000 20 python bin/draw.py -i 20 -t 3600 --biasOptimal --contextual -R 2400 --trainset=S12 --dopruning --skiptesting -a 2 --taskReranker split --taskSplit 0,11,31

#bash run.sh S12.5.3 10000 20 python bin/draw.py -i 20 -t 450 --biasOptimal --contextual -R 450 --trainset=S12 --dopruning --skiptesting -a 2 --taskReranker split --taskSplit 10,17,30
#bash run.sh S12.5.4 10000 20 python bin/draw.py -i 20 -t 450 --biasOptimal --contextual -R 450 --trainset=S12 --dopruning --skiptesting -a 2 --taskReranker split --taskSplit 10,17,30
#bash run.sh S12.5.5 10000 20 python bin/draw.py -i 20 -t 450 --biasOptimal --contextual -R 450 --trainset=S12 --dopruning --skiptesting -a 2 --taskReranker split --taskSplit 10,17,30

#bash run.sh S12.7 15000 20 python bin/draw.py -i 20 -t 3600 --biasOptimal --contextual -R 2400 --trainset=S12 --dopruning --skiptesting -a 2 --taskReranker split --taskSplit 10,17,30
#bash run.sh S12.8 10000 20 python bin/draw.py -i 20 -t 1440 --biasOptimal --contextual -R 1800 --trainset=S12 --dopruning --skiptesting -a 2 --taskReranker split --taskSplit 10,17,30
#bash run.sh S13.8 10000 20 python bin/draw.py -i 20 -t 1440 --biasOptimal --contextual -R 1800 --trainset=S13 --dopruning --skiptesting -a 2 --taskReranker split --taskSplit 10,17,30

#bash run.sh S12.9 15000 20 python bin/draw.py -i 20 -t 3600 --biasOptimal --contextual -R 2400 --trainset=S12 --dopruning --testingTimeout 720 -a 2 --taskReranker split --taskSplit 10,30
#bash run.sh S13.9 15000 20 python bin/draw.py -i 20 -t 3600 --biasOptimal --contextual -R 2400 --trainset=S13 --dopruning --testingTimeout 720 -a 2 --taskReranker split --taskSplit 10,30

#bash run.sh S12.5.6 5000 20 python bin/draw.py -i 20 -t 450 --biasOptimal --contextual -R 450 --trainset=S12 --dopruning --skiptesting -a 2 --taskReranker unsolved

#bash run.sh S12.8.1 5000 20 python bin/draw.py -i 20 -t 1440 --biasOptimal --contextual -R 1800 --trainset=S12 --dopruning --skiptesting -a 2 --taskReranker unsolved

#bash run.sh S12.8.2 5000 20 python bin/draw.py -i 20 -t 1440 --biasOptimal --contextual -R 1800 --trainset=S12 --dopruning --skiptesting -a 2 --taskReranker default --taskBatchSize 11

# bash run.sh S12.10 8000 20 python bin/draw.py -i 20 -t 3600 --biasOptimal --contextual -R 2400 --trainset=S12 --dopruning --testingTimeout 1440 -a 2 --taskReranker unsolved
# bash run.sh S13.10 8000 20 python bin/draw.py -i 20 -t 3600 --biasOptimal --contextual -R 2400 --trainset=S13 --dopruning --testingTimeout 1440 -a 2 --taskReranker unsolved

bash run.sh S12.10.test 10000 20 python bin/draw.py -i 25 -t 10 --biasOptimal --contextual -R 2400 --trainset=S12 --dopruning --testingTimeout 7200 -a 2 --taskReranker unsolved --resume experimentOutputs/draw/2019-11-05T10:59:39.638549/draw_aic=1.0_arity=2_BO=True_CO=True_dp=True_doshaping=False_ES=1_ET=3600_HR=0.5_it=14_MF=5_noConsol=False_pc=30_RT=2400_RR=False_RW=False_st=False_solver=ocaml_STM=True_L=1.5_TRR=unsolved_K=2_topkNotMAP=False_tset=S12.pickle
bash run.sh S13.10.test 10000 20 python bin/draw.py -i 25 -t 10 --biasOptimal --contextual -R 2400 --trainset=S13 --dopruning --testingTimeout 7200 -a 2 --taskReranker unsolved --resume experimentOutputs/draw/2019-11-05T10:59:39.800347/draw_aic=1.0_arity=2_BO=True_CO=True_dp=True_doshaping=False_ES=1_ET=3600_HR=0.5_it=18_MF=5_noConsol=False_pc=30_RT=2400_RR=False_RW=False_st=False_solver=ocaml_STM=True_L=1.5_TRR=unsolved_K=2_topkNotMAP=False_tset=S13.pickle

# draw_aic=1.0_arity=2_BO=True_CO=True_dp=True_doshaping=False_ES=1_ET=10_HR=0.5_it=14_MF=5_noConsol=False_pc=30_RT=2400_RR=False_RW=False_st=False_solver=ocaml_STM=True_L=1.5_TRR=unsolved_K=2_topkNotMAP=False_tset=S12.pickle


# bash run.sh S13.10 8000 20 python bin/draw.py -i 20 -t 3600 --biasOptimal --contextual -R 2400 --trainset=S13 --dopruning --testingTimeout 1440 -a 2 --taskReranker unsolved

# bash run.sh S12.10.test 8000 20 python bin/draw.py -i 20 -t 10 --biasOptimal --contextual -R 2400 --trainset=S12 --dopruning --testingTimeout 3600 -a 2 --taskReranker unsolved --checkpoint experimentOutputs/draw_aic=1.0_arity=2_BO=True_CO=True_dp=True_doshaping=False_ES=1_ET=3600_HR=0.5_it=14_MF=5_noConsol=False_pc=30_RT=2400_RR=False_RW=False_st=False_solver=ocaml_STM=True_L=1.5_TRR=unsolved_K=2_topkNotMAP=False_tset=S12.pickle
