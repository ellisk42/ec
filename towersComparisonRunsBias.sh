#runs to do:

#checkpoint=

time=1
testingTime=1200
recSteps=480000 #list repl is roughly 1k/hour (0.33 steps/sec)
ncores=8
#salt=towers
helmRatio=0.5
#resume=experimentOutputs/towers



for num in 3 20
	do
		#symbolic

		#replPolicy=experimentOutputs/towers${num}LongREPL.pickle
		filterMotifs="brickBaseInvention brickBaseReverse oddLoops oddMoves"
		oldResume=experimentOutputs/towers${num}
		samplePolicy=experimentOutputs/towers${num}BiasSample.pickle

		resume=experimentOutputs/towers${num}Bias
		salt=towers${num}Bias
		cp ${oldResume}.pickle ${resume}.pickle

		#rm ${resume}Symbolic.pickle_RecModelOnly
		# cp ${resume}.pickle ${resume}Symbolic.pickle
		# #cp ${oldResume}Sample.pickle_RecModelOnly ${resume}Symbolic.pickle_RecModelOnly
		# # #Train:
		# cmd="python bin/tower.py --searchType Astar --filterMotifs ${filterMotifs} --split 0.0 --tasks maxHard --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue Symbolic -i 2 --resume ${resume}Symbolic.pickle --singleRoundValueEval --seed 1"
		# #eval "${cmd}"
		# sbatch -e towersSymbolic${salt}.out -o towersSymbolic${salt}.out execute_gpu_new.sh ${cmd}
	
		#rm ${resume}Sample.pickle_RecModelOnly
		cp ${resume}.pickle ${resume}Sample.pickle
		#cp ${oldResume}Sample.pickle_RecModelOnly ${resume}Sample.pickle_RecModelOnly
		#Train:
		cmd="python bin/tower.py  --searchType Astar --filterMotifs ${filterMotifs} --split 0.0 --tasks maxHard --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue Sample -i 2 --resume ${resume}Sample.pickle --singleRoundValueEval --seed 2"
		#eval "${cmd}"
		sbatch -e towersSample${salt}.out -o towersSample${salt}.out execute_gpu_new.sh ${cmd}


		#REPL
		#rm ${resume}REPL.pickle_RecModelOnly
		cp ${resume}.pickle ${resume}REPL.pickle
		#cp ${oldResume}REPL.pickle_RecModelOnly ${resume}REPL.pickle_RecModelOnly
		#Train:
		cmd="python bin/tower.py  --searchType Astar --filterMotifs ${filterMotifs} --split 0.0 --tasks maxHard --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue TowerREPL -i 2  --resume ${resume}REPL.pickle --singleRoundValueEval --seed 2"
		sbatch -e towersREPL${salt}.out -o towersREPL${salt}.out execute_gpu_new.sh ${cmd}
		#eval "${cmd}"


		#RNN
		#rm ${resume}RNN.pickle_RecModelOnly
		cp ${resume}.pickle ${resume}RNN.pickle
		#cp ${oldResume}RNN.pickle_RecModelOnly ${resume}RNN.pickle_RecModelOnly
		cmd="python bin/tower.py  --searchType Astar --filterMotifs ${filterMotifs} --split 0.0 --tasks maxHard --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue RNN -i 2 --resume ${resume}RNN.pickle --singleRoundValueEval --seed 2"
		sbatch -e towersRNN${salt}.out -o towersRNN${salt}.out execute_gpu_new.sh ${cmd}


		# resume=experimentOutputs/towers${num}JustHashing
		# salt=towers${num}JustHashing
		# cp ${oldResume}.pickle ${resume}RNN.pickle
		# cp ${oldResume}RNN.pickle_RecModelOnly ${resume}RNN.pickle_RecModelOnly
		# cmd="python bin/tower.py --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} --primitives new --split 0.5 -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue RNN -i 2 --resume ${resume}RNN.pickle --singleRoundValueEval --seed 1"
		# sbatch -e towersRNN${salt}.out -o towersRNN${salt}.out execute_gpu_new.sh ${cmd}


		#eval "${cmd}"

	done