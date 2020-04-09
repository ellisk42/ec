#runs to do:

#checkpoint=

time=300
recSteps=240000 #list repl is roughly 1k/hour (0.33 steps/sec)
ncores=8
salt=towers
helmRatio=0.5
resume=experimentOutputs/towers

cp ${resume}.pickle ${resume}Sample.pickle
#Train:
sbatch -e towersSample${salt}.out -o towersSample${salt}.out execute_gpu_new.sh python bin/tower.py --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new --split 0.5 -t ${time} -RS ${recSteps} --solver 'python'  -c ${ncores} --useValue Sample -i 2 --resume ${resume}Sample.pickle --singleRoundValueEval

cp ${resume}.pickle ${resume}REPL.pickle
#Train:
om-repeat sbatch -e towersREPL${salt}.out -o towersREPL${salt}.out -p tenenbaum --time=3600 --mem=64G --cpus-per-task=8 --gres=gpu:QUADRORTX6000:1 python bin/tower.py --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new --split 0.5 -t ${time} -RS ${recSteps} --solver 'python'  -c ${ncores} --useValue TowerREPL -i 2  --resume ${resume}REPL.pickle --singleRoundValueEval

cp ${resume}.pickle ${resume}RNN.pickle
#Train:
sbatch -e towersRNN${salt}.out -o towersRNN${salt}.out execute_gpu_new.sh python bin/tower.py --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new --split 0.5 -t ${time} -RS ${recSteps} --solver 'python'  -c ${ncores} --useValue RNN -i 2 --resume ${resume}RNN.pickle --singleRoundValueEval

#omitted --resumeTraining for this run

#experimentOutputs/towers/2020-04-09T09:43:51.288294/tower_aic=1.0_arity=3_BO=False_CO=False_ES=1_ET=2_HR=0.5_it=2_MF=5_noCons=True_pc=10_RS=1000_RT=3600_resTrain=False_RR=False_RW=False_SRVE=False_sTr=False_solver=python_STM=True_L=1_TRR=default_K=2_topkNotMAP=False_useVal=TowerREPL_graph=True.pickle