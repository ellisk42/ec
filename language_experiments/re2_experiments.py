singularity_base_command = "srun --job-name=re2_language_{} --output=jobs/{} --ntasks=1 --mem-per-cpu=5000 --gres=gpu --cpus-per-task 24 --time=10000:00 --qos=tenenbaum --partition=tenenbaum singularity exec -B /om2  --nv ../dev-container.img "

experiment_commands = []
jobs = []
job = 0

# Generates EC baseline experiments
RUN_EC_BASELINES = False
for enumerationTimeout in [720, 1800]:
    job_name = "re2_ec_learned_feature_compression_et_{}".format(enumerationTimeout)
    jobs.append(job_name)
    
    num_iterations = 5
    task_batch_size = 25
    test_every = 8 # Every 200 tasks 
    base_parameters = "--no-cuda --enumerationTimeout {} --testingTimeout 720 --recognitionEpochs 10 --biasOptimal --contextual --Helmholtz 0 --iterations {} --taskBatchSize {} --testEvery {}".format(enumerationTimeout, num_iterations, task_batch_size, test_every)
    
    base_command = "python bin/re2.py "
    
    singularity = singularity_base_command.format(job, job_name)
    command = singularity + base_command + base_parameters + " &"
    if RUN_EC_BASELINES:
        experiment_commands.append(command)
    job += 1

# Generates restricted letter count baseline experiments
RUN_EC_RESTRICTED_BASELINES = True
for dataset in ['re2_500_aesdrt', 're2_500_aesr']:
    for enumerationTimeout in [720, 1800]:
        job_name = "re2_ec_learned_feature_compression_et_{}_{}".format(enumerationTimeout, dataset)
        jobs.append(job_name)
        
        num_iterations = 5
        task_batch_size = 20
        test_every = 8 # Every 200 tasks 
        primitives = 're2_4_letter' if dataset == 're2_500_aesr' else 're2_6_letter'
        base_parameters = "--no-cuda --enumerationTimeout {} --testingTimeout 720 --recognitionEpochs 10 --biasOptimal --contextual --Helmholtz 0 --iterations {} --taskBatchSize {} --testEvery {} --taskDataset {} --primitives {}".format(enumerationTimeout, num_iterations, task_batch_size, test_every, dataset, primitives)
        
        base_command = "python bin/re2.py "
        
        singularity = singularity_base_command.format(job, job_name)
        command = singularity + base_command + base_parameters + " &"
        if RUN_EC_RESTRICTED_BASELINES:
            experiment_commands.append(command)
        job += 1

# Generate Helmholtz generative model experiments.

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
