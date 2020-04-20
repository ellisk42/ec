singularity_base_command = "srun --job-name=logo_language_{} --output=jobs/{} --ntasks=1 --mem-per-cpu=20000 --gres=gpu --cpus-per-task 24 --time=10000:00 --qos=tenenbaum --partition=tenenbaum singularity exec -B /om2  --nv ../dev-container.img "

experiment_commands = []
jobs = []
job = 0

# Generates EC baseline experiments
RUN_EC_BASELINES = False
for enumerationTimeout in [1800, 3600]:
    # TODO (@CathyWong) -- these parameters are outdated as of 4/9/2020
    job_name = "logo_ec_cnn_compression_et_{}".format(enumerationTimeout)
    jobs.append(job_name)
    
    num_iterations = 5
    task_batch_size = 40
    test_every = 3 # Every 120 tasks 
    base_parameters = "--no-cuda --enumerationTimeout {} --testingTimeout {} --recognitionEpochs 10 --biasOptimal --contextual --Helmholtz 0.5 --iterations {} --taskBatchSize {} --testEvery {}".format(enumerationTimeout, enumerationTimeout, num_iterations, task_batch_size, test_every)
    
    base_command = "python bin/logo.py "
    
    singularity = singularity_base_command.format(job, job_name)
    command = singularity + base_command + base_parameters + " &"
    if RUN_EC_BASELINES:
        experiment_commands.append(command)
    job += 1
for enumerationTimeout in [1800, 3600]:
    job_name = "logo_ec_compression_et_{}".format(enumerationTimeout)
    jobs.append(job_name)
    
    num_iterations = 5
    task_batch_size = 40
    test_every = 3 # Every 120 tasks 
    base_parameters = "--no-recognition --no-cuda --enumerationTimeout {} --testingTimeout {} --recognitionEpochs 10 --biasOptimal --contextual --Helmholtz 0.5 --iterations {} --taskBatchSize {} --testEvery {}".format(enumerationTimeout, enumerationTimeout, num_iterations, task_batch_size, test_every)
    
    base_command = "python bin/logo.py "
    
    singularity = singularity_base_command.format(job, job_name)
    command = singularity + base_command + base_parameters + " &"
    if RUN_EC_BASELINES:
        experiment_commands.append(command)
    job += 1
for enumerationTimeout in [1800, 3600]:
    job_name = "logo_ec_cnn_et_{}".format(enumerationTimeout)
    jobs.append(job_name)
    
    num_iterations = 5
    task_batch_size = 40
    test_every = 3 # Every 120 tasks 
    base_parameters = "--no-consolidation --no-cuda --enumerationTimeout {} --testingTimeout {} --recognitionEpochs 10 --biasOptimal --contextual --Helmholtz 0.5 --iterations {} --taskBatchSize {} --testEvery {}".format(enumerationTimeout, enumerationTimeout, num_iterations, task_batch_size, test_every)
    
    base_command = "python bin/logo.py "
    
    singularity = singularity_base_command.format(job, job_name)
    command = singularity + base_command + base_parameters + " &"
    if RUN_EC_BASELINES:
        experiment_commands.append(command)
    job += 1

#### Generates language search baseline experiments.
RUN_LANGUAGE_SEARCH_BASELINE = False

enumerationTimeout = 1800
num_iterations = 5
task_batch_size = 40
test_every = 3 # Every 120 tasks 
# Language only: Enumerate, recognition_1 = language, compression
job_name = "logo_ec_gru_compression_et_{}".format(enumerationTimeout)
jobs.append(job_name)
base_command = "python bin/logo.py "
base_parameters = f"--enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --language_encoder recurrent --taskDataset logo_unlimited_200 --languageDataset logo_unlimited_200/synthetic "
exp_parameters = "--recognitionEpochs 100 --recognition_0 --recognition_1 language --Helmholtz 0"

singularity = singularity_base_command.format(job, job_name)
command = singularity + base_command + base_parameters + exp_parameters + " &"
if RUN_LANGUAGE_SEARCH_BASELINE:
    experiment_commands.append(command)
job += 1

# Separate recognition: Enumerate, recognition_0 = examples, recognition_1 = language, compression
job_name = "logo_ec_cnn_gru_compression_et_{}".format(enumerationTimeout)
exp_parameters = " --recognitionTimeout 3600 --recognition_0 examples --recognition_1 language --Helmholtz 0.5"
singularity = singularity_base_command.format(job, job_name)
command = singularity + base_command + base_parameters + exp_parameters + " &"
if RUN_LANGUAGE_SEARCH_BASELINE:
    experiment_commands.append(command)
job += 1

# Finetune with language: Enumerate, recognition_0 = examples, recognition_1 = examples, language, compression
job_name = "logo_ec_cnn_gru_cnn_compression_et_{}".format(enumerationTimeout)
exp_parameters = " --recognitionTimeout 3600 --recognition_0 examples --recognition_1 examples language --Helmholtz 0.5 --finetune_1"
singularity = singularity_base_command.format(job, job_name)
command = singularity + base_command + base_parameters + exp_parameters + " &"
if RUN_LANGUAGE_SEARCH_BASELINE:
    experiment_commands.append(command)
job += 1

# Label Helmholtz with nearest.
job_name = "logo_ec_cnn_gru_cnn_nearest_compression_et_{}".format(enumerationTimeout)
exp_parameters = " --recognitionTimeout 3600 --recognition_0 examples --recognition_1 examples language --Helmholtz 0.5 --finetune_1  --helmholtz_nearest_language 1"
singularity = singularity_base_command.format(job, job_name)
command = singularity + base_command + base_parameters + exp_parameters + " &"
if RUN_LANGUAGE_SEARCH_BASELINE:
    experiment_commands.append(command)
job += 1

#### Generates EC baselines with updated LOGO dataset and supervision
RUN_EC_BASELINES_LOGO_2 = True
enumerationTimeout = 1800
num_iterations = 12
task_batch_size = 40
test_every = 3
for dataset in ['logo_unlimited_200', 'logo_unlimited_500', 'logo_unlimited_1000']:
    for sample_n_supervised in [0, 10, 15]:
        job_name = f"logo_2_ec_cnn_compression_et_{enumerationTimeout}_supervised_{sample_n_supervised}"
        jobs.append(job_name)
        base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionTimeout 1800 --recognition_0 examples --Helmholtz 0.5"
        exp_parameters = f" --taskDataset {dataset} --sample_n_supervised {sample_n_supervised}"
        singularity = singularity_base_command.format(job, job_name)
        command = singularity + base_command + base_parameters + exp_parameters + " &"
        if RUN_EC_BASELINES_LOGO_2:
            experiment_commands.append(command)
        job +=1
#### Generate Helmholtz generative model experiments.
RUN_HELMHOLTZ_GENERATIVE_MODEL = True
enumerationTimeout = 1800
num_iterations = 12
task_batch_size = 40
test_every = 3
for dataset in ['logo_unlimited_200', 'logo_unlimited_500', 'logo_unlimited_1000']:
    for sample_n_supervised in [0, 10, 15]:
        for phrase_length in [1, 3, 7]:
            job_name = f"logo_2_ec_cnn_gru_ghelm_compression_et_{enumerationTimeout}_supervised_{sample_n_supervised}"
            jobs.append(job_name)
            base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionTimeout 1800 --recognition_1 examples language --Helmholtz 0.5 --induce_synchronous_grammar"
            exp_parameters = f" --taskDataset {dataset} --language_encoder recurrent --languageDataset {dataset}/synthetic --sample_n_supervised {sample_n_supervised} --moses_dir ./moses_compiled"
            singularity = singularity_base_command.format(job, job_name)
            command = singularity + base_command + base_parameters + exp_parameters + " &"
            if RUN_HELMHOLTZ_GENERATIVE_MODEL:
                experiment_commands.append(command)
            job +=1
            


#### Outputs
PRINT_LOG_SCRIPT = False
PRINT_JOBS = True
if PRINT_JOBS and not PRINT_LOG_SCRIPT:
    # print the jobs.
    print('#!/bin/bash')
    print("module add openmind/singularity")
    for command in experiment_commands:
        print(command + "")
        
if PRINT_LOG_SCRIPT:
    for job_name in jobs:
        print("echo 'Job: jobs/{} '".format(job_name))
        print("echo 'Training tasks:' ".format(job_name))
        print("grep 'total hit tasks' jobs/{}".format(job_name))
        print("echo 'Testing tasks:' ".format(job_name))
        print("grep 'testing tasks' jobs/{}".format(job_name))
        # Error checking
        print("grep 'OSError' jobs/{}".format(job_name))
        print("grep 'slurmstepd' jobs/{}".format(job_name))