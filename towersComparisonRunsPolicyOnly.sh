#runs to do:

#checkpoint=

time=1
testingTime=2000
recSteps=480000 #list repl is roughly 1k/hour (0.33 steps/sec)
ncores=8
#salt=towers
helmRatio=0.5
#resume=experimentOutputs/towers



for num in 3 20
	do
		#symbolic

		oldResume=experimentOutputs/towers${num}


		resume=experimentOutputs/towers${num}PolicyOnly
		salt=towers${num}PolicyOnly
		cp ${oldResume}.pickle ${resume}.pickle


		#REPL
		#rm ${resume}REPL.pickle_RecModelOnly
		cp ${resume}.pickle ${resume}REPL.pickle
		#
		#Train:
		#--initializePolicyWithValueWeights ${REPLValue}
		cmd="python bin/tower.py --policyType REPL --searchType SMC --split 0.0 --tasks maxHard --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue Sample -i 2  --resume ${resume}REPL.pickle --singleRoundValueEval --seed 2"
	
		sbatch -e towersREPL${salt}.out -o towersREPL${salt}.out execute_gpu_new.sh ${cmd}

		#RNN
		cp ${resume}.pickle ${resume}RNN.pickle
		cmd="python bin/tower.py --policyType RNN --searchType SMC --split 0.0 --tasks maxHard --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue Sample -i 2 --resume ${resume}RNN.pickle --singleRoundValueEval --seed 2"
		sbatch -e towersRNN${salt}.out -o towersRNN${salt}.out execute_gpu_new.sh ${cmd}

	done