#runs to do:

#checkpoint=

time=1
testingTime=1
recSteps=10000 #list repl is roughly 1k/hour (0.33 steps/sec)
ncores=8
#salt=towers
helmRatio=0.5
#resume=experimentOutputs/towers



for num in 3 #20
	do
		#symbolic

		#replPolicy=experimentOutputs/towers${num}LongREPL.pickle
		#filterMotifs="brickBaseInvention brickBaseReverse oddLoops oddMoves"
		oldResume=experimentOutputs/towers${num}
		#samplePolicy=experimentOutputs/towers${num}BiasSample.pickle
		RNNValue=experimentOutputs/towers${num}RNN.pickle
		REPLValue=experimentOutputs/towers${num}REPL.pickle


		resume=experimentOutputs/towers${num}Finetune
		salt=towers${num}Finetune
		cp ${oldResume}.pickle ${resume}.pickle


		#REPL
		#rm ${resume}REPL.pickle_RecModelOnly
		cp ${resume}.pickle ${resume}REPL.pickle
		#
		#Train:
		#--initializePolicyWithValueWeights ${REPLValue}
		cmd="python bin/tower.py --policyType REPL --searchType SMC --split 0.0 --tasks maxHard --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue Sample -i 2  --resume ${resume}REPL.pickle --singleRoundValueEval --seed 2"
		eval "${cmd}"
		cp ${resume}REPL.pickle_RecModelOnly ${resume}NoFTREPL.pickle_RecModelOnly
		cp ${resume}REPL_SRE=True.pickle ${resume}NoFTRNREPL_SRE=True.pickle
		cp ${resume}REPL_SRE=True_graph=True.pickle ${resume}NoFTRNREPL_SRE=True_graph=True.pickle

		cmd="python bin/tower.py --policyType REPL --initializePolicyWithValueWeights ${REPLValue} --searchType SMC --split 0.0 --tasks maxHard --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue Sample -i 2  --resume ${resume}REPL.pickle --singleRoundValueEval --seed 2"
		#sbatch -e towersREPL${salt}.out -o towersREPL${salt}.out execute_gpu_new.sh ${cmd}
		eval "${cmd}"


		#RNN
		#rm ${resume}RNN.pickle_RecModelOnly
		cp ${resume}.pickle ${resume}RNN.pickle
		#cp ${oldResume}RNN.pickle_RecModelOnly ${resume}RNN.pickle_RecModelOnly
		#--initializePolicyWithValueWeights ${RNNValue}
		cmd="python bin/tower.py --policyType RNN --searchType SMC --split 0.0 --tasks maxHard --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue Sample -i 2 --resume ${resume}RNN.pickle --singleRoundValueEval --seed 2"
		eval "${cmd}"
		cp ${resume}RNN.pickle_RecModelOnly ${resume}NoFTRNN.pickle_RecModelOnly
		cp ${resume}RNN_SRE=True.pickle ${resume}NoFTRNRNN_SRE=True.pickle
		cp ${resume}RNN_SRE=True_graph=True.pickle ${resume}NoFTRNRNN_SRE=True_graph=True.pickle

		cmd="python bin/tower.py --initializePolicyWithValueWeights ${RNNValue} --policyType RNN --searchType SMC --split 0.0 --tasks maxHard --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue Sample -i 2 --resume ${resume}RNN.pickle --singleRoundValueEval --seed 2"
		eval "${cmd}"
		#sbatch -e towersRNN${salt}.out -o towersRNN${salt}.out execute_gpu_new.sh ${cmd}


		# resume=experimentOutputs/towers${num}JustHashing
		# salt=towers${num}JustHashing
		# cp ${oldResume}.pickle ${resume}RNN.pickle
		# cp ${oldResume}RNN.pickle_RecModelOnly ${resume}RNN.pickle_RecModelOnly
		# cmd="python bin/tower.py --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new --split 0.5 -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue RNN -i 2 --resume ${resume}RNN.pickle --singleRoundValueEval --seed 1"
		# sbatch -e towersRNN${salt}.out -o towersRNN${salt}.out execute_gpu_new.sh ${cmd}


		#eval "${cmd}"

	done