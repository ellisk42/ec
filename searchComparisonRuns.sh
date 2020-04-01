#runs to do:

#checkpoint='experimentOutputs/list/2020-03-19T12:53:04.711372/list_aic=1.0_arity=3_BO=False_CO=False_ES=1_ET=20_HR=0.5_it=1_MF=10_noConsolidation=False_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_solver=python_STM=True_L=1.0_TRR=default_K=2_topkNotMAP=False_useValue=AbstractREPL.pickle'

time=300
recSteps=480000 #repl is roughly 1k/hour (0.33 steps/sec)
ncores=8
salt=richPrimsR1
#resume=experimentOutputs/listBaseIT=1 #experimentOutputs/listCathyTestEnum
#resume=experimentOutputs/listCathyTest
resume=experimentOutputs/listRichPrimsR1

cp ${resume}.pickle ${resume}Sample.pickle
#Train:
sbatch -e listSample${salt}.out -o listSample${salt}.out execute_gpu_new.sh python bin/list.py --recognitionTimeout 216000 --resumeTraining -r 1 --primitives rich --split 0.5 -t ${time} -RS ${recSteps} --solver 'python'  -c ${ncores} --useValue Sample -i 2 -H 512 --resume ${resume}Sample.pickle --singleRoundValueEval &

cp ${resume}.pickle ${resume}REPL.pickle
#Train:
sbatch -e listREPL${salt}.out -o listREPL${salt}.out execute_gpu_new.sh python bin/list.py --recognitionTimeout 216000 --resumeTraining -r 1 --primitives rich --split 0.5 -t ${time} -RS ${recSteps} --solver 'python'  -c ${ncores} --useValue AbstractREPL -i 2 -H 512 --resume ${resume}REPL.pickle --singleRoundValueEval &

cp ${resume}.pickle ${resume}RNN.pickle
#Train:
sbatch -e listRNN${salt}.out -o listRNN${salt}.out execute_gpu_new.sh python bin/list.py --recognitionTimeout 216000 --resumeTraining -r 1 --primitives rich --split 0.5 -t ${time} -RS ${recSteps} --solver 'python'  -c ${ncores} --useValue RNN -i 2 -H 512 --resume ${resume}RNN.pickle --singleRoundValueEval &

#omitted --resumeTraining for this run