#runs to do:

#checkpoint=

time=600
testingTime=600
recSteps=240000 #480000 #list repl is roughly 1k/hour (0.33 steps/sec)
ncores=8
#salt=towers
helmRatio=0.5
#resume=experimentOutputs/towers

for num in 1 3 20
	do
		resume=experimentOutputs/towers${num}
		salt=towers${num}EvalSamplePolicy
		cp ${resume}.pickle ${resume}Sample.pickle
		#Train:
		cmd="python bin/tower.py --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new --split 0.5 -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue Sample -i 2 --resume ${resume}Sample.pickle --singleRoundValueEval --seed 1"
		#eval "${cmd}"
		#sbatch -e towersSample${salt}.out -o towersSample${salt}.out execute_gpu_new.sh ${cmd}

		cp ${resume}.pickle ${resume}REPL.pickle
		#Train:
		cmd="python bin/tower.py --useSamplePolicy ${resume}Sample.pickle --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new --split 0.5 -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue TowerREPL -i 2  --resume ${resume}REPL.pickle --singleRoundValueEval --seed 1"
		#om-repeat sbatch -e towersREPL${salt}.out -o towersREPL${salt}.out -p tenenbaum --time=3600 --mem=32G --cpus-per-task=8 --gres=gpu:QUADRORTX6000:1 ${cmd}
		sbatch -e towersREPL${salt}.out -o towersREPL${salt}.out execute_gpu_new.sh ${cmd}
		#eval "${cmd}"

		cp ${resume}.pickle ${resume}RNN.pickle
		#Train:
		cmd="python bin/tower.py --useSamplePolicy ${resume}Sample.pickle --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new --split 0.5 -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue RNN -i 2 --resume ${resume}RNN.pickle --singleRoundValueEval --seed 1"
		#cmd="python bin/tower.py --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new --split 0.5 -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue RNN -i 2 --resume ${resume}RNN.pickle --singleRoundValueEval --seed 1"
		sbatch -e towersRNN${salt}.out -o towersRNN${salt}.out execute_gpu_new.sh ${cmd}
		#eval "${cmd}"

	done