#runs to do:

#checkpoint='experimentOutputs/list/2020-03-19T12:53:04.711372/list_aic=1.0_arity=3_BO=False_CO=False_ES=1_ET=20_HR=0.5_it=1_MF=10_noConsolidation=False_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_solver=python_STM=True_L=1.0_TRR=default_K=2_topkNotMAP=False_useValue=AbstractREPL.pickle'




sbatch -e listEnum.out -o listEnum.out execute_multicore.sh python bin/list.py --split 0.5 -t 1 -RS 200 --solver 'python'  -c 4 -i 7 -H 512 --resume experimentOutputs/listCathyTestEnum.pickle --singleRoundValueEval

sbatch -e listREPL.out -o listREPL.out execute_multicore.sh python bin/list.py --split 0.5 -t 1 -RS 200 --solver 'python'  -c 4 --useValue AbstractREPL -i 7 -H 512 --resume experimentOutputs/listCathyTestREPL.pickle --singleRoundValueEval

sbatch -e listRNN.out -o listRNN.out execute_multicore.sh python bin/list.py --split 0.5 -t 1 -RS 200 --solver 'python'  -c 4 --useValue RNN -i 7 -H 512 --resume experimentOutputs/listCathyTestRNN.pickle --singleRoundValueEval