#runs to do:

#checkpoint=

time=1
testingTime=600
recSteps=20000000 #list repl is roughly 1k/hour (0.33 steps/sec)
ncores=8
#salt=towers
helmRatio=1.0
#resume=experimentOutputs/towers



for num in ""
	do
		#symbolic

		resume=experimentOutputs/rb${num}PolicyOnly
		salt=PolicyOnly #Test600oldtasks
		#cp ${oldResume}.pickle ${resume}.pickle


		#Bigram
		#cp ${resume}.pickle ${resume}Bigram.pickle
		cmd="python bin/rb.py --dataset old --searchType Astar --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} -t ${time} -RS ${recSteps} --solver python -c ${ncores} --useValue Sample -i 2 --resume ${resume}Bigram.pickle  --singleRoundValueEval --seed 5"
		sbatch -e rbBigram${salt}.out -o rbBigram${salt}.out execute_gpu_new.sh ${cmd}
		#eval "${cmd}"

		#REPL
		#cp ${resume}.pickle ${resume}REPL.pickle
		cmd="python bin/rb.py --dataset old --policyType RBREPL --searchType Astar --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} -t ${time} -RS ${recSteps} --solver python  -c ${ncores} --useValue Sample -i 2  --resume ${resume}REPL.pickle --singleRoundValueEval --seed 5"
		sbatch -e rbREPL${salt}.out -o rbREPL${salt}.out execute_gpu_new.sh ${cmd}
		#eval "${cmd}"

		#RNN
		#cp ${resume}.pickle ${resume}RNN.pickle
		cmd="python bin/rb.py --dataset old --policyType RNN --searchType Astar --contextual --testingTimeout ${testingTime} --recognitionTimeout 216000 --resumeTraining -r ${helmRatio} -t ${time} -RS ${recSteps} --solver python -c ${ncores} --useValue Sample -i 2 --resume ${resume}RNN.pickle  --singleRoundValueEval --seed 5"
		sbatch -e rbRNN${salt}.out -o rbRNN${salt}.out execute_gpu_new.sh ${cmd}
		#eval "${cmd}"

	done
