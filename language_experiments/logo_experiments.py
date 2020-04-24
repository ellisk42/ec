USING_GCLOUD = False
USING_SINGULARITY = True
NUM_REPLICATIONS = 3

def gcloud_commands(job_name):
    gcloud_disk_command = f"gcloud compute --project 'tenenbaumlab' disks create {job_name} --size '30' --zone 'us-east1-b' --source-snapshot 'zyzzyva-logo-language' --type 'pd-standard'"
    gcloud_launch_commmand = f"gcloud beta compute --project=tenenbaumlab instances create {job_name} --zone=us-east1-b --machine-type=n1-standard-32 --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=150557817012-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --disk=name={job_name.strip()},device-name={job_name.strip()},mode=rw,boot=yes,auto-delete=yes --reservation-affinity=any"
    return f"#######\n{gcloud_disk_command}\n\n{gcloud_launch_commmand}\n\n###Now run: \nsingularity exec ../dev-container.img "

singularity_base_command = "srun --job-name=logo_language_{} --output=jobs/{} --ntasks=1 --mem-per-cpu=20000 --gres=gpu --cpus-per-task 20 --time=10000:00 --qos=tenenbaum --partition=tenenbaum singularity exec -B /om2  --nv ../dev-container.img "

def get_launcher_command(job, job_name):
    if USING_SINGULARITY:
        return singularity_base_command.format(job, job_name)
    else:
        job_name = f'logo-language-{job} '
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
        repl_name = f'_repl_{i}'
        repl_job = str(job) + repl_name
        repl_job_name = job_name + repl_name
        replication_command = f'--taskReranker randomShuffle --seed {i}'
        replications += [build_command(exp_command, repl_job, repl_job_name, replication_command)]
    return replications

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
    exp_command =  base_command + base_parameters
    command = build_command(exp_command, job, job_name, replication=None)
    if RUN_EC_BASELINES:
        experiment_commands.append(command)
        experiment_commands += build_replications(exp_command, job, job_name)
    job += 1
for enumerationTimeout in [1800, 3600]:
    job_name = "logo_ec_compression_et_{}".format(enumerationTimeout)
    jobs.append(job_name)
    
    num_iterations = 5
    task_batch_size = 40
    test_every = 3 # Every 120 tasks 
    base_parameters = "--no-recognition --no-cuda --enumerationTimeout {} --testingTimeout {} --recognitionEpochs 10 --biasOptimal --contextual --Helmholtz 0.5 --iterations {} --taskBatchSize {} --testEvery {}".format(enumerationTimeout, enumerationTimeout, num_iterations, task_batch_size, test_every)
    
    base_command = "python bin/logo.py "
    exp_command = base_command + base_parameters
    command = build_command(exp_command, job, job_name, replication=None)
    if RUN_EC_BASELINES:
        experiment_commands.append(command)
        experiment_commands += build_replications(exp_command, job, job_name)
    job += 1
for enumerationTimeout in [1800, 3600]:
    job_name = "logo_ec_cnn_et_{}".format(enumerationTimeout)
    jobs.append(job_name)
    
    num_iterations = 5
    task_batch_size = 40
    test_every = 3 # Every 120 tasks 
    base_parameters = "--no-consolidation --no-cuda --enumerationTimeout {} --testingTimeout {} --recognitionEpochs 10 --biasOptimal --contextual --Helmholtz 0.5 --iterations {} --taskBatchSize {} --testEvery {}".format(enumerationTimeout, enumerationTimeout, num_iterations, task_batch_size, test_every)
    
    base_command = "python bin/logo.py "
    
    exp_command = base_command + base_parameters
    command = build_command(exp_command, job, job_name, replication=None) 
    if RUN_EC_BASELINES:
        experiment_commands.append(command)
        experiment_commands += build_replications(exp_command, job, job_name)
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

exp_command = base_command + base_parameters + exp_parameters
command = build_command(exp_command, job, job_name, replication=None)

if RUN_LANGUAGE_SEARCH_BASELINE:
    experiment_commands.append(command)
    experiment_commands += build_replications(exp_command, job, job_name)
job += 1

# Separate recognition: Enumerate, recognition_0 = examples, recognition_1 = language, compression
job_name = "logo_ec_cnn_gru_compression_et_{}".format(enumerationTimeout)
exp_parameters = " --recognitionTimeout 3600 --recognition_0 examples --recognition_1 language --Helmholtz 0.5"

exp_command = base_command + base_parameters + exp_parameters
command = build_command(exp_command, job, job_name, replication=None)
if RUN_LANGUAGE_SEARCH_BASELINE:
    experiment_commands.append(command)
    experiment_commands += build_replications(exp_command, job, job_name)
job += 1

# Finetune with language: Enumerate, recognition_0 = examples, recognition_1 = examples, language, compression
job_name = "logo_ec_cnn_gru_cnn_compression_et_{}".format(enumerationTimeout)
exp_parameters = " --recognitionTimeout 3600 --recognition_0 examples --recognition_1 examples language --Helmholtz 0.5 --finetune_1"
command = get_launcher_command(job, job_name) + base_command + base_parameters + exp_parameters + append_command(job_name)
if RUN_LANGUAGE_SEARCH_BASELINE:
    experiment_commands.append(command)
    experiment_commands += build_replications(exp_command, job, job_name)
job += 1

# Label Helmholtz with nearest.
job_name = "logo_ec_cnn_gru_cnn_nearest_compression_et_{}".format(enumerationTimeout)
exp_parameters = " --recognitionTimeout 3600 --recognition_0 examples --recognition_1 examples language --Helmholtz 0.5 --finetune_1  --helmholtz_nearest_language 1"
exp_command = base_command + base_parameters + exp_parameters
command = build_command(exp_command, job, job_name, replication=None)
if RUN_LANGUAGE_SEARCH_BASELINE:
    experiment_commands.append(command)
    experiment_commands += build_replications(exp_command, job, job_name)
job += 1

#### Generates EC baselines with updated LOGO dataset and supervision
RUN_EC_BASELINES_LOGO_2 = True
enumerationTimeout = 1800
num_iterations = 12
task_batch_size = 40
test_every = 3
recognition_timeout = 1800
EXPS = [('logo_unlimited_200', 0)]
for dataset in ['logo_unlimited_200', 'logo_unlimited_500', 'logo_unlimited_1000']:
    for sample_n_supervised in [0, 10]:
        exp = (dataset, sample_n_supervised)
        job_name = f"logo_2_ec_cnn_compression_et_{enumerationTimeout}_supervised_{sample_n_supervised}_{dataset}"
        jobs.append(job_name)
        base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionTimeout {recognition_timeout} --recognition_0 examples --Helmholtz 0.5"
        exp_parameters = f" --taskDataset {dataset} --sample_n_supervised {sample_n_supervised}"
        
        exp_command = base_command + base_parameters + exp_parameters
        command = build_command(exp_command, job, job_name, replication=None)
        
        if RUN_EC_BASELINES_LOGO_2:
            if (EXPS is None) or (exp in EXPS):
                experiment_commands.append(command)
                experiment_commands += build_replications(exp_command, job, job_name)
        job +=1
#### Generate Helmholtz generative model experiments.
RUN_HELMHOLTZ_GENERATIVE_MODEL = False
EXPS = [('logo_unlimited_200', 0, 1), ('logo_unlimited_200', 10, 1)]
enumerationTimeout = 1800
num_iterations = 12
task_batch_size = 40
test_every = 3
recognition_timeout = 1800
for dataset in ['logo_unlimited_200', 'logo_unlimited_500', 'logo_unlimited_1000']:
    for sample_n_supervised in [0, 10]:
        for phrase_length in [5,3,1]:
            exp = (dataset, sample_n_supervised, phrase_length)
            job_name = f"logo_2_ec_cnn_gru_ghelm_compression_et_{enumerationTimeout}_supervised_{sample_n_supervised}_{dataset}_pl_{phrase_length}"
            jobs.append(job_name)
            base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionTimeout {recognition_timeout} --recognition_0 --recognition_1 examples language --Helmholtz 0.5 --synchronous_grammar"
            exp_parameters = f" --taskDataset {dataset} --language_encoder recurrent --languageDataset {dataset}/synthetic --sample_n_supervised {sample_n_supervised} --moses_dir ./moses_compiled --smt_phrase_length {phrase_length}"
        
            exp_command = base_command + base_parameters + exp_parameters
            command = build_command(exp_command, job, job_name, replication=None)
            if RUN_HELMHOLTZ_GENERATIVE_MODEL:
                if (EXPS is None) or (exp in EXPS):
                    experiment_commands.append(command)
                    experiment_commands += build_replications(exp_command, job, job_name)
            job +=1


#### Generate Helmholtz pseudoalignment experiments.
RUN_HELMHOLTZ_PSEUDOALIGNMENTS = False
enumerationTimeout = 1800
num_iterations = 12
task_batch_size = 40
test_every = 3
recognition_steps = 10000
EXPS = [('logo_unlimited_200', 0, 1), ('logo_unlimited_200', 10, 1)]
pseudoalignment = 0.1
for dataset in ['logo_unlimited_200', 'logo_unlimited_500', 'logo_unlimited_1000']:
    for sample_n_supervised in [0, 10]:
        for phrase_length in [5,3,1]:
            exp = (dataset, sample_n_supervised, phrase_length)
            job_name = f"logo_2_ec_cnn_gru_ghelm_pseudo_compression_et_{enumerationTimeout}_supervised_{sample_n_supervised}_{dataset}_pl_{phrase_length}"
            jobs.append(job_name)
            base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionSteps {recognition_steps} --recognition_0 --recognition_1 examples language --Helmholtz 0.5 --synchronous_grammar"
            exp_parameters = f" --taskDataset {dataset} --language_encoder recurrent --languageDataset {dataset}/synthetic --sample_n_supervised {sample_n_supervised} --moses_dir ./moses_compiled --smt_phrase_length {phrase_length} --smt_pseudoalignments {pseudoalignment}"
            
            exp_command = base_command + base_parameters + exp_parameters
            command = build_command(exp_command, job, job_name, replication=None)
            if RUN_HELMHOLTZ_PSEUDOALIGNMENTS:
                if (EXPS is None) or (exp in EXPS):
                    experiment_commands.append(command)
                    experiment_commands += build_replications(exp_command, job, job_name)
            job +=1

RUN_NO_HELMHOLTZ_GENERATIVE_MODEL = False
enumerationTimeout = 1800
num_iterations = 12
task_batch_size = 40
test_every = 3
recognition_steps = 10000
EXPS = [('logo_unlimited_200', 0)]
for dataset in ['logo_unlimited_200', 'logo_unlimited_500', 'logo_unlimited_1000']:
    for sample_n_supervised in [0, 10]:
        exp = (dataset, sample_n_supervised)
        job_name = f"logo_2_ec_cnn_gru_no_ghelm_compression_et_{enumerationTimeout}_supervised_{sample_n_supervised}_{dataset}"
        jobs.append(job_name)
        base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionSteps {recognition_steps} --recognition_0 --recognition_1 examples language --Helmholtz 0 --skip_first_test"
        exp_parameters = f" --taskDataset {dataset} --language_encoder recurrent --languageDataset {dataset}/synthetic --sample_n_supervised {sample_n_supervised} "
    
        command = get_launcher_command(job, job_name) + base_command + base_parameters + exp_parameters + append_command(job_name)
        if RUN_NO_HELMHOLTZ_GENERATIVE_MODEL:
            if (EXPS is None) or (exp in EXPS):
                experiment_commands.append(command)
                experiment_commands += build_replications(exp_command, job, job_name)
        job +=1

#### Outputs
PRINT_LOG_SCRIPT = False
PRINT_JOBS = True
if PRINT_JOBS and not PRINT_LOG_SCRIPT:
    # print the jobs.
    print('#!/bin/bash')
    if USING_SINGULARITY:
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