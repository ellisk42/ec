#run JoshRULE:


for i in {0..49}
	do 
		sbatch -o predictions_${i}.csv -e samples_${i}.csv execute_cpu_josh.sh python bin/rbBaseline.py -d 'josh' --test josh_wave_3.p -w 3 --timeout 600 --tasks $i
	done