# Generates language experiment scripts.

base_language_dataset = "data/logo/language/307FVKVSYSP6M7F60SONLCHBLTN74V_3_4_2020_gen_4/nwords_None_gen_{}_clean.json"
num_iterations=10
featureExtractors = ["ngram", "recurrent"]

singularity_base_command = "srun --job-name=logo_language_{} --output=jobs/{} --ntasks=1 --mem-per-cpu=5000 --gres=gpu --cpus-per-task 15 --time=10000:00 --qos=tenenbaum --partition=tenenbaum singularity exec -B /om2  --nv ../dev-container.img "

base_parameters = "--no-cuda --enumerationTimeout 3600 --testingTimeout 3600 --split 0.5 --recognitionEpochs 10 --biasOptimal --contextual --Helmholtz 0 --iterations {}".format(num_iterations)

### Experiments
experiment_commands = []
jobs = []
job = 0
# Log linear parser
base_command = "python bin/logo.py --parser loglinear --useWakeLanguage  --no-recognition --recognitionEpochs 10 "

for featureExtractor in featureExtractors:
    for language_gen in range(0, 4+1):
        job_name = "logo_language_log_linear_{}_gen_{}".format(featureExtractor, language_gen)
        jobs.append(job_name)
        
        languageDataset = base_language_dataset.format(language_gen)
        
        singularity = singularity_base_command.format(job, job_name)
        command = singularity + base_command + base_parameters + " --languageFeatureExtractor {}".format(featureExtractor) + " --languageDataset {}".format(languageDataset) + " &"
        experiment_commands.append(command)
        job += 1

# MLP Recognition model
base_command = "python bin/logo.py --recognitionEpochs 10 "
for featureExtractor in featureExtractors:
    for language_gen in range(0, 4+1):
        job_name = "logo_language_nn_{}_gen_{}".format(featureExtractor, language_gen)
        jobs.append(job_name)
        
        languageDataset = base_language_dataset.format(language_gen)
        singularity = singularity_base_command.format(job, job_name)
        command = singularity + base_command + base_parameters + " --languageFeatureExtractor {}".format(featureExtractor) + " --languageDataset {}".format(languageDataset) + " &"
        experiment_commands.append(command)
        job += 1

if False:
    # print the jobs.
    print('#!/bin/bash')
    print("module add openmind/singularity")
    for command in experiment_commands:
        print(command + "")
if True:
    for job_name in jobs:
        print("echo 'Job: jobs/{} '".format(job_name))
        print("echo 'Training tasks:' ".format(job_name))
        print("grep 'total hit tasks' jobs/{}".format(job_name))
        print("echo 'Testing tasks:' ".format(job_name))
        print("grep 'testing tasks' jobs/{}".format(job_name))

        
