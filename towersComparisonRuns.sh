#runs to do:

#checkpoint=

time=1
testingTime=1200
recSteps=240000 #list repl is roughly 1k/hour (0.33 steps/sec)
ncores=8
#salt=towers
helmRatio=0.5
#resume=experimentOutputs/towers



for num in 20
	do
		#symbolic

		cp experimentOutputs/towers${num}.pickle experimentOutputs/towers${num}MaxTasks.pickle

		replPolicy=experimentOutputs/towers${num}LongREPL.pickle
		samplePolicy=experimentOutputs/towers${num}Sample.pickle
		oldResume=experimentOutputs/towers${num}

		resume=experimentOutputs/towers${num}MaxTasks
		
		salt=towers${num}MaxTasks
		cp ${resume}.pickle ${resume}Symbolic.pickle

		cp ${oldResume}Sample.pickle_RecModelOnly ${resume}Symbolic.pickle_RecModelOnly
		#Train:
		cmd="python bin/tower.py --tasks maxHard --split 0.0 --useSamplePolicy ${samplePolicy} --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue Symbolic -i 2 --resume ${resume}Symbolic.pickle --singleRoundValueEval --seed 1"
		#eval "${cmd}"
		sbatch -e towersSymbolic${salt}.out -o towersSymbolic${salt}.out execute_gpu_new.sh ${cmd}

		#sample
		#oldResume=${resume}

		cp ${resume}.pickle ${resume}Sample.pickle
		cp ${oldResume}Sample.pickle_RecModelOnly ${resume}Sample.pickle_RecModelOnly
		#Train:
		cmd="python bin/tower.py --tasks maxHard --split 0.0 --useSamplePolicy ${samplePolicy} --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue Sample -i 2 --resume ${resume}Sample.pickle --singleRoundValueEval --seed 1"
		#eval "${cmd}"
		sbatch -e towersSample${salt}.out -o towersSample${salt}.out execute_gpu_new.sh ${cmd}

		#REPL
		cp ${replPolicy} ${resume}REPL.pickle
		cp ${replPolicy}_RecModelOnly ${resume}REPL.pickle_RecModelOnly
		#Train:
		#cmd="python bin/tower.py --useSamplePolicy experimentOutputs/towers${num}Sample.pickle --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new --split 0.5 -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue TowerREPL -i 2  --resume ${resume}REPL.pickle --singleRoundValueEval --seed 1"
		#cmd="python bin/tower.py --useSamplePolicy ${resume}Sample.pickle --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new --split 0.5 -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue TowerREPL -i 2  --resume ${resume}REPL.pickle --singleRoundValueEval --seed 1"
		cmd="python bin/tower.py --tasks maxHard --split 0.0 --useSamplePolicy ${samplePolicy} --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue TowerREPL -i 2  --resume ${resume}REPL.pickle --singleRoundValueEval --seed 1"
		#om-repeat sbatch -e towersREPL${salt}.out -o towersREPL${salt}.out -p tenenbaum --time=3600 --mem=32G --cpus-per-task=8 --gres=gpu:QUADRORTX6000:1 ${cmd}
		sbatch -e towersREPL${salt}.out -o towersREPL${salt}.out execute_gpu_new.sh ${cmd}
		#eval "${cmd}"

		#RNN
		cp ${resume}.pickle ${resume}RNN.pickle
		cp ${oldResume}RNN.pickle_RecModelOnly ${resume}RNN.pickle_RecModelOnly
		cmd="python bin/tower.py --tasks maxHard --split 0.0 --useSamplePolicy ${samplePolicy} --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue RNN -i 2 --resume ${resume}RNN.pickle --singleRoundValueEval --seed 1"
		sbatch -e towersRNN${salt}.out -o towersRNN${salt}.out execute_gpu_new.sh ${cmd}


		# resume=experimentOutputs/towers${num}JustHashing
		# salt=towers${num}JustHashing
		# cp ${oldResume}.pickle ${resume}RNN.pickle
		# cp ${oldResume}RNN.pickle_RecModelOnly ${resume}RNN.pickle_RecModelOnly
		# cmd="python bin/tower.py --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new --split 0.5 -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue RNN -i 2 --resume ${resume}RNN.pickle --singleRoundValueEval --seed 1"
		# sbatch -e towersRNN${salt}.out -o towersRNN${salt}.out execute_gpu_new.sh ${cmd}


		#eval "${cmd}"

	done