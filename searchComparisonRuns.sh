#runs to do:

#checkpoint='experimentOutputs/list/2020-03-19T12:53:04.711372/list_aic=1.0_arity=3_BO=False_CO=False_ES=1_ET=20_HR=0.5_it=1_MF=10_noConsolidation=False_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_solver=python_STM=True_L=1.0_TRR=default_K=2_topkNotMAP=False_useValue=AbstractREPL.pickle'

time=300
recSteps=480000 #repl is roughly 10k/hour (0.33 steps/sec)
ncores=36
salt=richPrims
#resume=experimentOutputs/listBaseIT=1 #experimentOutputs/listCathyTestEnum
#resume=experimentOutputs/listCathyTest
resume=experimentOutputs/listRichPrims

cp ${resume}.pickle ${resume}Sample.pickle
sbatch -e listSample${salt}.out -o listSample${salt}.out execute_multicore.sh python bin/list.py --primitives rich --split 0.5 -t ${time} -RS ${recSteps} --solver 'python'  -c ${ncores} --useValue Sample -i 2 -H 512 --resume ${resume}Sample.pickle --singleRoundValueEval &

cp ${resume}.pickle ${resume}REPL.pickle
sbatch -e listREPL${salt}.out -o listREPL${salt}.out execute_multicore.sh python bin/list.py --primitives rich --split 0.5 -t ${time} -RS ${recSteps} --solver 'python'  -c ${ncores} --useValue AbstractREPL -i 2 -H 512 --resume ${resume}REPL.pickle --singleRoundValueEval &

cp ${resume}.pickle ${resume}RNN.pickle
sbatch -e listRNN${salt}.out -o listRNN${salt}.out execute_multicore.sh python bin/list.py --primitives rich --split 0.5 -t ${time} -RS ${recSteps} --solver 'python'  -c ${ncores} --useValue RNN -i 2 -H 512 --resume ${resume}RNN.pickle --singleRoundValueEval &


# messing about:
#'experimentOutputs/listCathyTestIT=1.pickle'

#python bin/list.py --split 0.5 -t ${time} -RS ${recSteps} --solver 'python'  -c ${ncores} --useValue AbstractREPL -i 8 -H 512 --resume 'experimentOutputs/listCathyTestIT=1.pickle' --singleRoundValueEval


#one good nocompuression run for getting some candidates:
#switch to master, then use multiple cpus and run:
#nohup python bin/list.py --split 0.5 -t 1200 --solver 'ocaml' -c 20 -i 1 --no-recognition --no-consolidation &> base_run.txt &
#then cp to experimentOutputs/listBaseIT=1.pickle


#sbatch -e listSampleCathy.out -o listSampleCathy.out execute_multicore.sh python bin/list.py --split 0.5 -t 300 -RS 2000 --solver 'python'  -c 20 --useValue Sample -i 2 -H 512 --resume listCathyTestSample.pickle --singleRoundValueEval &