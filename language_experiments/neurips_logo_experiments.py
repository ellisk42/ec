USING_AZURE = True
USING_GCLOUD = False
USING_SINGULARITY = False
NUM_REPLICATIONS = 3
NO_ORIGINAL_REPL = False

LANGUAGE_COMPRESSION = False

# Base images.
AZURE_IMAGES = [
    ("ec-language-5-24",  "ec-language-east2",  "--location eastus2"),
    ("ec-language-compression-5-25", "ec-language-east2",  "--location eastus2"),
    ("ec-language-embeddings-5-25", "ec-language-central", ""),
    ("ec-embeddings-central", "ec-language-central-us", ""),
    ("ec-embeddings-west", "ec-language-west", ""),
]
AZURE_IMAGE = ("ec-embeddings-central", "ec-language-central-us", "")
    

def azure_commands(job_name): 
    image_name, group, location = AZURE_IMAGE
    machine_type = "Standard_E48s_v3"
    azure_launch_command = f"az vm create --name az-{job_name} --resource-group {group} --generate-ssh-keys --data-disk-sizes-gb 128 --image {image_name} --size {machine_type} --public-ip-address-dns-name {job_name} {location} \n"
    
    return f"#######\n{azure_launch_command}###Now run: \n git pull; "

def gcloud_commands(job_name):
    gcloud_disk_command = f"gcloud compute --project 'tenenbaumlab' disks create {job_name} --size '100' --zone 'us-east1-b' --source-snapshot 'logo-language-april29' --type 'pd-standard'"
    gcloud_launch_commmand = f"gcloud beta compute --project=tenenbaumlab instances create {job_name} --zone=us-east1-b --machine-type=n1-highmem-64 --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=150557817012-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --disk=name={job_name.strip()},device-name={job_name.strip()},mode=rw,boot=yes,auto-delete=yes --reservation-affinity=any"
    return f"#######\n{gcloud_disk_command}\n\n{gcloud_launch_commmand}\n\n###Now run: \nsingularity exec ../dev-container.img "

singularity_base_command = "srun --job-name=logo_language_{} --output=jobs/{} --ntasks=1 --mem-per-cpu=20000 --gres=gpu --cpus-per-task 24 --time=10000:00 --qos=tenenbaum --partition=tenenbaum singularity exec -B /om2  --nv ../dev-container.img "

def get_launcher_command(job, job_name):
    if USING_SINGULARITY:
        return singularity_base_command.format(job, job_name)
    elif USING_GCLOUD:
        job_name = f'clevr-language-{job} '
        return gcloud_commands(job_name)
    else:
        job_name = f'clevr-language-{job} '
        return azure_commands(job_name)

def append_command(job_name):
    if USING_SINGULARITY:
        return  " &"
    else:
        return f"> jobs/{job_name} 2>&1 &"

def build_command(exp_command, job, job_name, replication=" "):
    if replication is None: replication = " "
    command = get_launcher_command(job, job_name) + exp_command + replication + append_command(job_name)
    if USING_GCLOUD:
        command = command.replace("python", "python3")
    if USING_AZURE:
        command = command.replace("python", "python3.7")
    return command

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
        if not NO_ORIGINAL_REPL: experiment_commands.append(command)
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
        if not NO_ORIGINAL_REPL: experiment_commands.append(command)
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
        if not NO_ORIGINAL_REPL: experiment_commands.append(command)
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
    if not NO_ORIGINAL_REPL: experiment_commands.append(command)
    experiment_commands += build_replications(exp_command, job, job_name)
job += 1

# Separate recognition: Enumerate, recognition_0 = examples, recognition_1 = language, compression
job_name = "logo_ec_cnn_gru_compression_et_{}".format(enumerationTimeout)
exp_parameters = " --recognitionTimeout 3600 --recognition_0 examples --recognition_1 language --Helmholtz 0.5"

exp_command = base_command + base_parameters + exp_parameters
command = build_command(exp_command, job, job_name, replication=None)
if RUN_LANGUAGE_SEARCH_BASELINE:
    if not NO_ORIGINAL_REPL: experiment_commands.append(command)
    experiment_commands += build_replications(exp_command, job, job_name)
job += 1

# Finetune with language: Enumerate, recognition_0 = examples, recognition_1 = examples, language, compression
job_name = "logo_ec_cnn_gru_cnn_compression_et_{}".format(enumerationTimeout)
exp_parameters = " --recognitionTimeout 3600 --recognition_0 examples --recognition_1 examples language --Helmholtz 0.5 --finetune_1"
command = get_launcher_command(job, job_name) + base_command + base_parameters + exp_parameters + append_command(job_name)
if RUN_LANGUAGE_SEARCH_BASELINE:
    if not NO_ORIGINAL_REPL: experiment_commands.append(command)
    experiment_commands += build_replications(exp_command, job, job_name)
job += 1

# Label Helmholtz with nearest.
job_name = "logo_ec_cnn_gru_cnn_nearest_compression_et_{}".format(enumerationTimeout)
exp_parameters = " --recognitionTimeout 3600 --recognition_0 examples --recognition_1 examples language --Helmholtz 0.5 --finetune_1  --helmholtz_nearest_language 1"
exp_command = base_command + base_parameters + exp_parameters
command = build_command(exp_command, job, job_name, replication=None)
if RUN_LANGUAGE_SEARCH_BASELINE:
    if not NO_ORIGINAL_REPL: experiment_commands.append(command)
    experiment_commands += build_replications(exp_command, job, job_name)
job += 1

#### Generates EC baselines with updated LOGO dataset and supervision
RUN_EC_BASELINES_LOGO_2 = False
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
        base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionTimeout {recognition_timeout} --recognition_0 examples --Helmholtz 0.5 --skip_first_test"
        exp_parameters = f" --taskDataset {dataset} --sample_n_supervised {sample_n_supervised}"
        
        exp_command = base_command + base_parameters + exp_parameters
        orig_command = exp_command + " --om_original_ordering 1 "
        command = build_command(orig_command, job, job_name, replication=None)
        
        if RUN_EC_BASELINES_LOGO_2:
            if (EXPS is None) or (exp in EXPS):
                if not NO_ORIGINAL_REPL: 
                    experiment_commands.append(command)
                experiment_commands += build_replications(exp_command, job, job_name)
        job +=1
#### Generate Helmholtz generative model experiments.
RUN_HELMHOLTZ_GENERATIVE_MODEL = False
EXPS = [('logo_unlimited_200', 10, 1)]
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
            base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionTimeout {recognition_timeout} --recognition_0 --recognition_1 examples language --Helmholtz 0.5 --synchronous_grammar --skip_first_test"
            exp_parameters = f" --taskDataset {dataset} --language_encoder recurrent --languageDataset {dataset}/synthetic --sample_n_supervised {sample_n_supervised} --moses_dir ./moses_compiled --smt_phrase_length {phrase_length}"
        
            exp_command = base_command + base_parameters + exp_parameters
            command = build_command(exp_command, job, job_name, replication=None)
            if RUN_HELMHOLTZ_GENERATIVE_MODEL:
                if (EXPS is None) or (exp in EXPS):
                    if not NO_ORIGINAL_REPL: experiment_commands.append(command)
                    experiment_commands += build_replications(exp_command, job, job_name)
            job +=1


#### Generate Helmholtz pseudoalignment experiments.
RUN_HELMHOLTZ_PSEUDOALIGNMENTS = False
enumerationTimeout = 1800
num_iterations = 12
task_batch_size = 40
test_every = 3
recognition_steps = 10000
EXPS = [('logo_unlimited_200', 0, 1)]
pseudoalignment = 0.1
for dataset in ['logo_unlimited_200', 'logo_unlimited_500', 'logo_unlimited_1000']:
    for sample_n_supervised in [0, 10]:
        for phrase_length in [5,3,1]:
            exp = (dataset, sample_n_supervised, phrase_length)
            job_name = f"logo_2_ec_cnn_gru_ghelm_pseudo_compression_et_{enumerationTimeout}_supervised_{sample_n_supervised}_{dataset}_pl_{phrase_length}"
            jobs.append(job_name)
            base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionSteps {recognition_steps} --recognition_0 --recognition_1 examples language --Helmholtz 0.5 --synchronous_grammar --skip_first_test"
            exp_parameters = f" --taskDataset {dataset} --language_encoder recurrent --languageDataset {dataset}/synthetic --sample_n_supervised {sample_n_supervised} --moses_dir ./moses_compiled --smt_phrase_length {phrase_length} --smt_pseudoalignments {pseudoalignment}"
            
            exp_command = base_command + base_parameters + exp_parameters
            orig_command = exp_command + " --om_original_ordering 1"
            command = build_command(orig_command, job, job_name, replication=None)
            if RUN_HELMHOLTZ_PSEUDOALIGNMENTS:
                if (EXPS is None) or (exp in EXPS):
                    if not NO_ORIGINAL_REPL: 
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
    
        exp_command = base_command + base_parameters + exp_parameters
        command = build_command(exp_command, job, job_name, replication=None)
        if RUN_NO_HELMHOLTZ_GENERATIVE_MODEL:
            if (EXPS is None) or (exp in EXPS):
                if not NO_ORIGINAL_REPL: experiment_commands.append(command)
                experiment_commands += build_replications(exp_command, job, job_name)
        job +=1

RUN_LANGUAGE_CURRICULUM_GHELM = False
EXPS = [('logo_unlimited_200', 0, 1)]
enumerationTimeout = 1800
num_iterations = 12
task_batch_size = 40
test_every = 3
recognition_timeout = 1800
for dataset in ['logo_unlimited_200', 'logo_unlimited_500', 'logo_unlimited_1000']:
    for sample_n_supervised in [0, 10]:
        for phrase_length in [5,3,1]:
            exp = (dataset, sample_n_supervised, phrase_length)
            job_name = f"logo_2_ec_cnn_gru_ghelm_compression_et_{enumerationTimeout}_supervised_{sample_n_supervised}_{dataset}_pl_{phrase_length}_curr"
            jobs.append(job_name)
            base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionTimeout {recognition_timeout} --recognition_0 --recognition_1 examples language --Helmholtz 0.5 --synchronous_grammar --skip_first_test --taskReranker sentence_length"
            exp_parameters = f" --taskDataset {dataset} --language_encoder recurrent --languageDataset {dataset}/synthetic --sample_n_supervised {sample_n_supervised} --moses_dir ./moses_compiled --smt_phrase_length {phrase_length}"
        
            exp_command = base_command + base_parameters + exp_parameters
            command = build_command(exp_command, job, job_name, replication=None)
            if RUN_LANGUAGE_CURRICULUM_GHELM:
                if (EXPS is None) or (exp in EXPS):
                    if not NO_ORIGINAL_REPL: experiment_commands.append(command)
                    experiment_commands += build_replications(exp_command, job, job_name)
            job +=1

RUN_LANGUAGE_CURRICULUM_BASELINE = False
enumerationTimeout = 1800
num_iterations = 12
task_batch_size = 40
test_every = 3
recognition_timeout = 1800
EXPS = [('logo_unlimited_200', 0)]
for dataset in ['logo_unlimited_200', 'logo_unlimited_500', 'logo_unlimited_1000']:
    for sample_n_supervised in [0, 10]:
        exp = (dataset, sample_n_supervised)
        job_name = f"logo_2_ec_cnn_compression_et_{enumerationTimeout}_supervised_{sample_n_supervised}_{dataset}_curr"
        jobs.append(job_name)
        base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionTimeout {recognition_timeout} --recognition_0 examples --Helmholtz 0.5 --skip_first_test --taskReranker sentence_length"
        exp_parameters = f" --taskDataset {dataset} --sample_n_supervised {sample_n_supervised} --languageDataset {dataset}/synthetic"
        
        
        exp_command = base_command + base_parameters + exp_parameters
        command = build_command(exp_command, job, job_name, replication=None)
    
        if RUN_LANGUAGE_CURRICULUM_BASELINE:
            if (EXPS is None) or (exp in EXPS):
                if not NO_ORIGINAL_REPL: experiment_commands.append(command)
                experiment_commands += build_replications(exp_command, job, job_name)
        job +=1

# Generate no-language test time experiments. For now, these are manual.
RUN_NO_LANGUAGE_TEST_EXPERIMENTS = False
language_checkpoints = [("best-ghelm-logo-language-18", "experimentOutputs/logo/2020-04-21T15-41-24-093831/logo_aic=1.0_arity=3_ET=1800_it=34_MF=5_n_models=1_no_dsl=F_pc=30.0_RS=10000_RW=F_STM=T_L=1.5_batch=40_K=2_topLL=F_LANG=F.pickle", 34, "logo_unlimited_200", "language"),
("best-baseline-logo-language-10", "experimentOutputs/logo/2020-04-20T00-39-24-721507/logo_aic=1.0_arity=3_BO=T_CO=T_ES=1_ET=1800_HR=0.5_it=27_MF=5_n_models=1_no_dsl=F_pc=30.0_RT=1800_RR=F_RW=F_smt_phrase_length=5_STM=T_L=1.5_synchronous_grammar=F_batch=40_K=2_topLL=F_LANG=F.pickle", 27, "logo_unlimited_200", "no-language")
] 
enumerationTimeout = 1800
recognition_timeout = 1800
for (name, language_checkpoint, last_iter, dataset, type) in language_checkpoints:
    job_name = f"logo_2_ec_test_no_language_{name}"
    jobs.append(job_name)
    base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {last_iter + 1} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery 1 --no-cuda --recognitionTimeout {recognition_timeout} --recognition_0 examples --Helmholtz 0.5 --iterations_as_epochs 0 "
    exp_parameters = f" --taskDataset {dataset} --languageDataset {dataset}/synthetic --resume {language_checkpoint} --no-dsl --no-consolidation --test_dsl_only "
    if type == 'language': exp_parameters += "--test_only_after_recognition" # Runs both tests.
    
    exp_command = base_command + base_parameters + exp_parameters
    command = build_command(exp_command, job, job_name, replication=None)
    if RUN_NO_LANGUAGE_TEST_EXPERIMENTS:
        if not NO_ORIGINAL_REPL: experiment_commands.append(command)
        experiment_commands += build_replications(exp_command, job, job_name)
    job +=1


#### Generate Helmholtz generative model experiments with language in the compression.
RUN_HELMHOLTZ_GENERATIVE_MODEL_LANGUAGE_COMPRESSION = False
EXPS = [('logo_unlimited_200', 0, 1)]
enumerationTimeout = 1800
num_iterations = 12
task_batch_size = 40
test_every = 3
recognition_timeout = 1800
lc_score = 0.2
max_compression = 5
for dataset in ['logo_unlimited_200', 'logo_unlimited_500', 'logo_unlimited_1000']:
    for sample_n_supervised in [0, 10]:
        for phrase_length in [5,3,1]:
            exp = (dataset, sample_n_supervised, phrase_length)
            job_name = f"logo_2_ec_cnn_gru_ghelm_lang_compression_et_{enumerationTimeout}_supervised_{sample_n_supervised}_{dataset}_pl_{phrase_length}"
            jobs.append(job_name)
            base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionTimeout {recognition_timeout} --recognition_0 --recognition_1 examples language --Helmholtz 0.5 --synchronous_grammar --skip_first_test"
            exp_parameters = f" --taskDataset {dataset} --language_encoder recurrent --languageDataset {dataset}/synthetic --sample_n_supervised {sample_n_supervised} --moses_dir ./moses_compiled --smt_phrase_length {phrase_length} --language_compression --lc_score {lc_score} --max_compression {max_compression} --om_original_ordering 1 "
        
            exp_command = base_command + base_parameters + exp_parameters
            command = build_command(exp_command, job, job_name, replication=None)
            if RUN_HELMHOLTZ_GENERATIVE_MODEL_LANGUAGE_COMPRESSION:
                if (EXPS is None) or (exp in EXPS):
                    if not NO_ORIGINAL_REPL: experiment_commands.append(command)
                    experiment_commands += build_replications(exp_command, job, job_name)
            job +=1

#### Generate Helmholtz generative model experiments with language in the compression.
RUN_HELMHOLTZ_GENERATIVE_MODEL_PSEUDO_LANGUAGE_COMPRESSION = False
EXPS = [('logo_unlimited_200', 0, 1)]
enumerationTimeout = 1800
num_iterations = 12
task_batch_size = 40
test_every = 3
recognition_timeout = 1800
lc_score = 0.2
max_compression = 5
pseudoalignment = 0.1
for dataset in ['logo_unlimited_200', 'logo_unlimited_500', 'logo_unlimited_1000']:
    for sample_n_supervised in [0, 10]:
        for phrase_length in [5,3,1]:
            exp = (dataset, sample_n_supervised, phrase_length)
            job_name = f"logo_2_ec_cnn_gru_ghelm_lang_compression_et_{enumerationTimeout}_supervised_{sample_n_supervised}_{dataset}_pl_{phrase_length}"
            jobs.append(job_name)
            base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionTimeout {recognition_timeout} --recognition_0 --recognition_1 examples language --Helmholtz 0.5 --synchronous_grammar --skip_first_test"
            exp_parameters = f" --taskDataset {dataset} --language_encoder recurrent --languageDataset {dataset}/synthetic --sample_n_supervised {sample_n_supervised} --moses_dir ./moses_compiled --smt_phrase_length {phrase_length} --language_compression --lc_score {lc_score} --max_compression {max_compression} --om_original_ordering 1 --smt_pseudoalignments {pseudoalignment} "
        
            exp_command = base_command + base_parameters + exp_parameters
            command = build_command(exp_command, job, job_name, replication=None)
            if RUN_HELMHOLTZ_GENERATIVE_MODEL_PSEUDO_LANGUAGE_COMPRESSION:
                if (EXPS is None) or (exp in EXPS):
                    if not NO_ORIGINAL_REPL: experiment_commands.append(command)
                    experiment_commands += build_replications(exp_command, job, job_name)
            job +=1

#### Generate Helmholtz generative model experiments with language in the compression and human language.
RUN_HELMHOLTZ_HUMAN_LANGUAGE_COMPRESSION = False
EXPS = [('logo_unlimited_200', 0, 0.5), ('logo_unlimited_200', 0.1, 0.5), ('logo_unlimited_200', 0, 0)]
enumerationTimeout = 1800
num_iterations = 12
task_batch_size = 40
test_every = 3
recognition_timeout = 1800
lc_score = 0.2
max_compression = 5
sample_n_supervised = 0
for dataset in ['logo_unlimited_200', 'logo_unlimited_500', 'logo_unlimited_1000']:
    for pseudoalignment in [0, 0.1]:
        for helmholtz in [0, 0.5]:
            exp = (dataset, pseudoalignment, helmholtz)
            pseudo_name = "pseudo_" if pseudoalignment > 0 else ""
            no_ghelm = "no_" if helmholtz == 0 else ""
            job_name = f"logo_2_ec_cnn_gru_{no_ghelm}ghelm_{pseudo_name}lang_compression_et_{enumerationTimeout}_supervised_{sample_n_supervised}_{dataset}_pl_{phrase_length}_humans"
            jobs.append(job_name)
            
            language_compression = " --language_compression" if helmholtz > 0 else ""
            base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionSteps {recognition_steps} --recognition_0 --recognition_1 examples language --Helmholtz {helmholtz} --synchronous_grammar --skip_first_test"
            exp_parameters = f" --taskDataset {dataset} --language_encoder recurrent --languageDataset {dataset}/humans --sample_n_supervised {sample_n_supervised} --moses_dir ./moses_compiled --smt_phrase_length {phrase_length} --smt_pseudoalignments {pseudoalignment} {language_compression} --lc_score {lc_score} --max_compression {max_compression} --om_original_ordering 1 "
        
            exp_command = base_command + base_parameters + exp_parameters
            command = build_command(exp_command, job, job_name, replication=None)
            if RUN_HELMHOLTZ_HUMAN_LANGUAGE_COMPRESSION:
                if (EXPS is None) or (exp in EXPS):
                    if not NO_ORIGINAL_REPL: experiment_commands.append(command)
                    experiment_commands += build_replications(exp_command, job, job_name)
            job +=1

# Generates multimodal synthesis baselines with no DSL learning.
RUN_HELMHOLTZ_LANGUAGE_NO_COMPRESSION = True
EXPS = [('logo_unlimited_200', 0, 1)]
enumerationTimeout = 1800
num_iterations = 12
task_batch_size = 40
test_every = 3
recognition_timeout = 1800
lc_score = 0.2
max_compression = 5
pseudoalignment = 0.1
for dataset in ['logo_unlimited_200', 'logo_unlimited_500', 'logo_unlimited_1000']:
    for sample_n_supervised in [0, 10]:
        for phrase_length in [5,3,1]:
            exp = (dataset, sample_n_supervised, phrase_length)
            job_name = f"logo_2_ec_cnn_gru_no_ghelm_lang_compression_et_{enumerationTimeout}_supervised_{sample_n_supervised}_{dataset}_pl_{phrase_length}"
            jobs.append(job_name)
            base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionTimeout {recognition_timeout} --recognition_0 --recognition_1 examples language --Helmholtz 0  --skip_first_test"
            exp_parameters = f" --taskDataset {dataset} --language_encoder recurrent --languageDataset {dataset}/synthetic --sample_n_supervised {sample_n_supervised} --moses_dir ./moses_compiled --smt_phrase_length {phrase_length} --language_compression --lc_score {lc_score} --max_compression {max_compression} --om_original_ordering 1 --smt_pseudoalignments {pseudoalignment} --no-consolidation "
        
            exp_command = base_command + base_parameters + exp_parameters
            command = build_command(exp_command, job, job_name, replication=None)
            if RUN_HELMHOLTZ_LANGUAGE_NO_COMPRESSION:
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