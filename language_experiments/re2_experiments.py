USING_GCLOUD = False
USING_SINGULARITY = True
NUM_REPLICATIONS = 0
NO_ORIGINAL_REPL = False

def gcloud_commands(job_name):
    gcloud_disk_command = f"gcloud compute --project 'tenenbaumlab' disks create {job_name} --size '30' --zone 'us-east1-b' --source-snapshot 're2-language-april24' --type 'pd-standard'"
    gcloud_launch_commmand = f"gcloud beta compute --project=tenenbaumlab instances create {job_name} --zone=us-east1-b --machine-type=n1-highmem-64 --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=150557817012-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --disk=name={job_name.strip()},device-name={job_name.strip()},mode=rw,boot=yes,auto-delete=yes --reservation-affinity=any"
    return f"#######\n{gcloud_disk_command}\n\n{gcloud_launch_commmand}\n\n###Now run: \nsingularity exec ../dev-container.img "
    
singularity_base_command = "srun --job-name=re2_language_{} --output=jobs/{} --ntasks=1 --mem-per-cpu=10000 --gres=gpu --cpus-per-task 24 --time=10000:00 --qos=tenenbaum --partition=tenenbaum singularity exec -B /om2  --nv ../dev-container.img "

def get_launcher_command(job, job_name):
    if USING_SINGULARITY:
        return singularity_base_command.format(job, job_name)
    else:
        job_name = f're2-language-{job} '
        return gcloud_commands(job_name)

def append_command(job_name):
    if USING_SINGULARITY:
        return  " &"
    else:
        return f"> jobs/{job_name} 2>&1 &"

def build_command(exp_command, job, job_name, replication=" "):
    if replication is None: replication = " "
    return get_launcher_command(job, job_name) + exp_command + replication + append_command(job_name)

def build_replications(exp_command, job, job_name):
    replications = []
    for i in range(1, NUM_REPLICATIONS + 1):
        repl_name = f'_repl_{i}' if USING_SINGULARITY else f'-repl-{i}'
        repl_job = str(job) + repl_name
        repl_job_name = job_name + repl_name
        replication_command = f' --taskReranker randomShuffle --seed {i} '
        replications += [build_command(exp_command, repl_job, repl_job_name, replication_command)]
    return replications

experiment_commands = []
jobs = []
job = 0

# Generates EC baseline experiments -- version 1
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

# Generates restricted letter count baseline experiments -- version 1
RUN_EC_RESTRICTED_BASELINES = False
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

## Generates EC baseline experiments with the updated dataset
RUN_EC_BASELINES_2 = True
num_iterations = 5
task_batch_size = 40
test_every = 3
recognition_timeout = 1800
EXPS = None
for dataset in ['re2_1000', 're2_500_aeioubcdfgsrt']:
    for enumerationTimeout in [720]:
        for use_vowel in [True, False]:
            job_name = f"re_2_ec_no_lang_compression_et_{enumerationTimeout}_{dataset}_use_vowel_{use_vowel}"
            jobs.append(job_name)
            base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionTimeout {recognition_timeout} --recognition_0 examples --Helmholtz 0.5 --skip_first_test"
            
            restricted = 'aeioubcdfgsrt' 
            if restricted in dataset:
                primitives = f"re2_chars_{restricted} re2_bootstrap_v1_primitives"
            else:
                primitives = "re2_chars_None re2_bootstrap_v1_primitives"
            if use_vowel:
                primitives += " re2_vowel_consonant_primitives"
                
            exp_parameters = f" --taskDataset {dataset} --primitives {primitives} "
            exp_command = base_command + base_parameters + exp_parameters
            command = build_command(exp_command, job, job_name, replication=None)
            if RUN_EC_BASELINES_2:
                if (EXPS is None) or (exp in EXPS):
                    if not NO_ORIGINAL_REPL: experiment_commands.append(command)
                    experiment_commands += build_replications(exp_command, job, job_name)
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
