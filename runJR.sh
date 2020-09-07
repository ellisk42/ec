#run JoshRULE:


# #for wave 3:
# for i in {0..49}
#         do 
# 	        sbatch -o predictions_wave_3.csv -e samples_wave_3.csv execute_cpu_josh.sh python bin/rbBaseline.py -d 'josh' --test josh_wave_3.p -w 3 --timeout 600 --tasks ${i}
#         done


#for wave 3.1:run
# for i in {0..9}
# 	do 
# 		sbatch -o predictions_wave_31.csv -e samples_wave_31.csv execute_cpu_josh.sh python bin/rbBaseline.py -d 'josh' --test josh_wave_3.p -w 3.1 --timeout 600 --tasks ${i}
# 	done


for i in {65..69}
	do 
		sbatch -o predictions_wave_31_76.csv -e samples_wave_31_76.csv execute_cpu_josh.sh python bin/rbBaseline.py -d 'josh' --test josh_wave_3.p -w 3.1 --timeout 600 --tasks ${i} --specialtask
	done