USING_AZURE = True
USING_GCLOUD = False
USING_SINGULARITY = False
NUM_REPLICATIONS = 0
NO_ORIGINAL_REPL = False

LANGUAGE_COMPRESSION = False

# Base images.
AZURE_IMAGES = [
    ("ec-language-5-24",  "ec-language-east2",  "--location eastus2"),
    ("ec-language-compression-5-25", "ec-language-east2",  "--location eastus2"),
    ("ec-language-embeddings-5-25", "ec-language-central", ""),
    ("ec-embeddings-central", "ec-language-central-us", ""),
    ("ec-embeddings-us-west", "ec-language-west", ""),
]
AZURE_IMAGE = ("ec-embeddings-us-west", "ec-language-west", "")

def azure_commands(job_name): 
    image_name, group, location = AZURE_IMAGE
    machine_type = "Standard_D48s_v3"
    azure_launch_command = f"az vm create --name az-{job_name} --resource-group {group} --generate-ssh-keys --data-disk-sizes-gb 128 --image {image_name} --size {machine_type} --public-ip-address-dns-name {job_name} {location} \n"
    
    return f"#######\n{azure_launch_command}###Now run: \n git pull; "

def gcloud_commands(job_name):
    machine_type = 'm1-ultramem-40' if HIGH_MEM else 'n2-highmem-64'
    gcloud_disk_command = f"gcloud compute --project 'andreas-jacob-8fc0' disks create {job_name} --size '30' --zone 'us-east1-b' --source-snapshot 're2-language-5-14' --type 'pd-standard'"
    gcloud_launch_commmand = f"gcloud beta compute --project=andreas-jacob-8fc0 instances create {job_name} --metadata='startup-script=cd ec' --zone=us-east1-b --machine-type={machine_type} --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=project-service-account@andreas-jacob-8fc0.iam.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --disk=name={job_name.strip()},device-name={job_name.strip()},mode=rw,boot=yes,auto-delete=yes --reservation-affinity=any"
    return f"#######\n{gcloud_disk_command}\n\n{gcloud_launch_commmand}\n\n###Now run: \n "
    
singularity_base_command = "srun --job-name=re2_language_{} --output=jobs/{} --ntasks=1 --mem-per-cpu=15000 --gres=gpu --cpus-per-task 24 --time=10000:00 --qos=tenenbaum --partition=tenenbaum singularity exec -B /om2  --nv ../dev-container.img "

def get_launcher_command(job, job_name):
    if USING_SINGULARITY:
        return singularity_base_command.format(job, job_name)
    elif USING_GCLOUD:
        job_name = f're2-language-{job} '
        return gcloud_commands(job_name)
    else:
        job_name = f're2-language-{job} '
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
num_iterations = 10
task_batch_size = 40
test_every = 3
recognition_timeout = 1800
EXPS = [('re2_1000', 720, True)]
for dataset in ['re2_1000', 're2_500_aeioubcdfgsrt']:
    for enumerationTimeout in [720]:
        for use_vowel in [True, False]:
            exp = (dataset, enumerationTimeout, use_vowel)
            job_name = f"re_2_ec_no_lang_compression_et_{enumerationTimeout}_{dataset}_use_vowel_{use_vowel}"
            jobs.append(job_name)
            base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionTimeout {recognition_timeout} --recognition_0 examples --Helmholtz 0.5 --skip_first_test"
            
            # Which primitives set to use.
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

## Generates EC baseline experiments with the updated dataset
RUN_NO_HELMHOLTZ_GENERATIVE_MODEL = False
num_iterations = 10
task_batch_size = 40
test_every = 3
recognition_steps = 10000
EXPS = [('re2_1000', 720, False)]
for dataset in ['re2_1000', 're2_500_aeioubcdfgsrt']:
    for enumerationTimeout in [720, 1800]:
        testingTimeout = 720
        for use_vowel in [True, False]:
            exp = (dataset, enumerationTimeout, use_vowel)
            job_name = f"re_2_ec_gru_no_ghelm_compression_et_{enumerationTimeout}_{dataset}_use_vowel_{use_vowel}"
            jobs.append(job_name)
            base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {testingTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionSteps {recognition_steps} --recognition_0 --recognition_1 examples language --Helmholtz 0 --skip_first_test"
            
            # Which primitives set to use.
            restricted = 'aeioubcdfgsrt' 
            if restricted in dataset:
                primitives = f"re2_chars_{restricted} re2_bootstrap_v1_primitives"
            else:
                primitives = "re2_chars_None re2_bootstrap_v1_primitives"
            if use_vowel:
                primitives += " re2_vowel_consonant_primitives"
                
            exp_parameters = f" --taskDataset {dataset} --language_encoder recurrent --languageDataset {dataset}/synthetic --primitives {primitives} "
            exp_command = base_command + base_parameters + exp_parameters
            command = build_command(exp_command, job, job_name, replication=None)
            if RUN_NO_HELMHOLTZ_GENERATIVE_MODEL:
                if (EXPS is None) or (exp in EXPS):
                    if not NO_ORIGINAL_REPL: experiment_commands.append(command)
                    experiment_commands += build_replications(exp_command, job, job_name)
            job +=1

#### Generate Helmholtz generative model experiments.
RUN_HELMHOLTZ_GENERATIVE_MODEL = False
num_iterations = 10
task_batch_size = 40
test_every = 3
recognition_steps = 10000
testing_timeout = 720
EXPS = [('re2_1000', 720, False), ('re2_1000', 1800, False), ('re2_1000', 1800, True)]
for dataset in ['re2_1000', 're2_500_aeioubcdfgsrt']:
    for enumerationTimeout in [720, 1800]:
        for use_vowel in [True, False]:
            exp = (dataset, enumerationTimeout, use_vowel)
            job_name = f"re_2_ec_gru_ghelm_compression_et_{enumerationTimeout}_{dataset}_use_vowel_{use_vowel}"
            jobs.append(job_name)
            base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {testing_timeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionSteps {recognition_steps} --recognition_0 --recognition_1 examples language --Helmholtz 0.5 --synchronous_grammar --skip_first_test"
            
            # Which primitives set to use.
            restricted = 'aeioubcdfgsrt' 
            if restricted in dataset:
                primitives = f"re2_chars_{restricted} re2_bootstrap_v1_primitives"
            else:
                primitives = "re2_chars_None re2_bootstrap_v1_primitives"
            if use_vowel:
                primitives += " re2_vowel_consonant_primitives"
                
            exp_parameters = f" --taskDataset {dataset} --language_encoder recurrent --languageDataset {dataset}/synthetic --primitives {primitives} --moses_dir ./moses_compiled --smt_phrase_length 1"
            exp_command = base_command + base_parameters + exp_parameters
            command = build_command(exp_command, job, job_name, replication=None)
            if RUN_HELMHOLTZ_GENERATIVE_MODEL:
                if (EXPS is None) or (exp in EXPS):
                    if not NO_ORIGINAL_REPL: experiment_commands.append(command)
                    experiment_commands += build_replications(exp_command, job, job_name)
            job +=1
### Run Helmholtz pseudoalignments
RUN_HELMHOLTZ_PSEUDOALIGNMENTS = False
num_iterations = 10
task_batch_size = 40
test_every = 3
recognition_steps = 10000
EXPS = [('re2_1000', 720, False)]
pseudoalignment = 0.1
for dataset in ['re2_1000']:
    for enumerationTimeout in [720, 1800]:
        for use_vowel in [True, False]:
            exp = (dataset, enumerationTimeout, use_vowel)
            testing_timeout = 720
            job_name = f"re_2_ec_gru_ghelm_pseudo_compression_et_{enumerationTimeout}_{dataset}_use_vowel_{use_vowel}"
            jobs.append(job_name)
            base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {testing_timeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionSteps {recognition_steps} --recognition_0 --recognition_1 examples language --Helmholtz 0.5 --synchronous_grammar --skip_first_test"
            
            # Which primitives set to use.
            restricted = 'aeioubcdfgsrt' 
            if restricted in dataset:
                primitives = f"re2_chars_{restricted} re2_bootstrap_v1_primitives"
            else:
                primitives = "re2_chars_None re2_bootstrap_v1_primitives"
            if use_vowel:
                primitives += " re2_vowel_consonant_primitives"
                
            exp_parameters = f" --taskDataset {dataset} --language_encoder recurrent --languageDataset {dataset}/synthetic --primitives {primitives} --moses_dir ./moses_compiled --smt_phrase_length 1 --smt_pseudoalignments {pseudoalignment}"
            exp_command = base_command + base_parameters + exp_parameters
            command = build_command(exp_command, job, job_name, replication=None)
            if RUN_HELMHOLTZ_PSEUDOALIGNMENTS:
                if (EXPS is None) or (exp in EXPS):
                    if not NO_ORIGINAL_REPL: experiment_commands.append(command)
                    experiment_commands += build_replications(exp_command, job, job_name)
            job +=1
# Generate no-language test time experiments. For now, these are mnaual.
RUN_NO_LANGUAGE_TEST_EXPERIMENTS_GHELM = False
language_checkpoints = [("best-ghelm-re2-language-27-repl-2", "experimentOutputs/re2/2020-05-18T19-50-59-156776/re2_aic=1.0_arity=3_ET=720_it=30_MF=5_n_models=1_no_dsl=F_pc=30.0_RS=10000_RW=F_STM=T_L=10.0_batch=40_K=2_topLL=F_LANG=F.pickle", 30, 're2_1000', "language"),
] 

enumerationTimeout = 720
recognition_timeout = 1800
for (name, language_checkpoint, last_iter, dataset, type) in language_checkpoints:
    job_name = f"re2_ec_test_no_language_{name}"
    jobs.append(job_name)
    base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {last_iter + 1} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery 1 --no-cuda --recognitionTimeout {recognition_timeout} --recognition_0 examples --Helmholtz 0.5 --iterations_as_epochs 0 "
    exp_parameters = f" --taskDataset {dataset} --languageDataset {dataset}/synthetic --resume {language_checkpoint} --no-dsl --no-consolidation --test_dsl_only --primitives re2_chars_None re2_bootstrap_v1_primitives"
    if type == 'language': exp_parameters += "--test_only_after_recognition" # Runs both tests.
    
    exp_command = base_command + base_parameters + exp_parameters
    command = build_command(exp_command, job, job_name, replication=None)
    if RUN_NO_LANGUAGE_TEST_EXPERIMENTS_GHELM:
        if not NO_ORIGINAL_REPL: experiment_commands.append(command)
        experiment_commands += build_replications(exp_command, job, job_name)
    job +=1

# Run baseline DSL test.
RUN_BASELINE_DSL_ENUM_AND_RECOGNITION_TEST = False
enumerationTimeout = 720
recognition_timeout = 1800
dataset = 're2_1000'
job_name = f"re2_ec_test_baseline_dsl_et_{enumerationTimeout}"
jobs.append(job_name)
base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations 1 --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery 1 --no-cuda --recognitionTimeout {recognition_timeout} --recognition_0 examples --Helmholtz 1.0 --iterations_as_epochs 0 "
exp_parameters = f" --taskDataset {dataset} --languageDataset {dataset}/synthetic --no-dsl --no-consolidation --test_dsl_only --primitives re2_chars_None re2_bootstrap_v1_primitives --test_only_after_recognition" # Runs both tests.

exp_command = base_command + base_parameters + exp_parameters
command = build_command(exp_command, job, job_name, replication=None)
if RUN_BASELINE_DSL_ENUM_AND_RECOGNITION_TEST:
    if not NO_ORIGINAL_REPL: experiment_commands.append(command)
    experiment_commands += build_replications(exp_command, job, job_name)
job +=1


## Vowel baseline experiment.


##### Generates a host of language-guided experiments using the vowel primitive.
RUN_HELMHOLTZ_VOWEL_EXPERIMENTS = True
use_vowel = True
num_iterations = 10
task_batch_size = 40
test_every = 3
recognition_steps = 10000
testing_timeout = 720
lc_score = 0.2
max_compression = 5
EXPS = [
            ('re2_1000', 720, True, 0, 0, 0, True), # No compression
            ('re2_1000', 720, True, 0, 0, 0, False), # No generative
            ('re2_1000', 720, True, 0, 0.5, 0, False), # Generative language -- Helmholtz
            ('re2_1000', 720, True, 0.1, 0.5, 0, False), # Generative language + injectivity
            ('re2_1000', 720, True, 0.1, 0.5, 0.2, False), # Generative language + inject + lc
            
            
]
for dataset in ['re2_1000', 're2_500_aeioubcdfgsrt']:
    for enumerationTimeout in [720, 1800]:
        for use_vowel in [True, False]:
            for pseudoalignment in [0, 0.1]:
                for helmholtz in [0, 0.5]:
                    for lc_score in [0, 0.2]:
                        for no_consolidation in [True, False]:
                            exp = (dataset, enumerationTimeout, use_vowel, pseudoalignment, helmholtz, lc_score, no_consolidation)
                            pseudo_name = "pseudo_" if pseudoalignment > 0 else ""
                            no_ghelm = "no_" if helmholtz == 0 else ""
                            no_cons = "no_" if no_consolidation else ""
                            
                            job_name = f"re_2_ec_gru_{no_ghelm}ghelm_{pseudo_name}lang_{no_cons}compression_et_{enumerationTimeout}_{dataset}_use_vowel_{use_vowel}"
                            jobs.append(job_name)
                            base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {testing_timeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionSteps {recognition_steps} --recognition_0 --recognition_1 examples language --Helmholtz {helmholtz}  --skip_first_test "
                            if helmholtz > 0: 
                                base_parameters += " --synchronous_grammar "
                            
                            # Which primitives set to use.
                            restricted = 'aeioubcdfgsrt' 
                            if restricted in dataset:
                                primitives = f"re2_chars_{restricted} re2_bootstrap_v1_primitives"
                            else:
                                primitives = "re2_chars_None re2_bootstrap_v1_primitives"
                            if use_vowel:
                                primitives += " re2_vowel_consonant_primitives"
                                
                            exp_parameters = f" --taskDataset {dataset} --language_encoder recurrent --languageDataset {dataset}/synthetic --primitives {primitives} --moses_dir ./moses_compiled --smt_phrase_length 1 --smt_pseudoalignments {pseudoalignment}  --language_compression --lc_score {lc_score} --max_compression {max_compression} "
                            if no_consolidation:
                                exp_parameters += "--no-consolidation "
                            
                            exp_command = base_command + base_parameters + exp_parameters
                            command = build_command(exp_command, job, job_name, replication=None)
                            if RUN_HELMHOLTZ_VOWEL_EXPERIMENTS:
                                if (EXPS is None) or (exp in EXPS):
                                    if not NO_ORIGINAL_REPL: experiment_commands.append(command)
                                    experiment_commands += build_replications(exp_command, job, job_name)
                            job +=1
##### Generates the full vowel experiment using the human language dataset.
RUN_HELMHOLTZ_VOWEL_HUMAN_EXPERIMENTS = True
use_vowel = True
num_iterations = 10
task_batch_size = 40
test_every = 3
recognition_steps = 10000
testing_timeout = 720
lc_score = 0.2
max_compression = 5
EXPS = [
            ('re2_1000', 720, True, 0.1, 0.5, 0.2, False), # Generative language + inject + lc    
]
for dataset in ['re2_1000', 're2_500_aeioubcdfgsrt']:
    for enumerationTimeout in [720, 1800]:
        for use_vowel in [True, False]:
            for pseudoalignment in [0, 0.1]:
                for helmholtz in [0, 0.5]:
                    for lc_score in [0, 0.2]:
                        for no_consolidation in [True, False]:
                            exp = (dataset, enumerationTimeout, use_vowel, pseudoalignment, helmholtz, lc_score, no_consolidation)
                            pseudo_name = "pseudo_" if pseudoalignment > 0 else ""
                            no_ghelm = "no_" if helmholtz == 0 else ""
                            no_cons = "no_" if no_consolidation else ""
                            
                            job_name = f"re_2_ec_gru_{no_ghelm}ghelm_{pseudo_name}lang_{no_cons}compression_et_{enumerationTimeout}_{dataset}_use_vowel_{use_vowel}_human"
                            jobs.append(job_name)
                            base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout {testing_timeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size} --testEvery {test_every} --no-cuda --recognitionSteps {recognition_steps} --recognition_0 --recognition_1 examples language --Helmholtz {helmholtz}  --skip_first_test --pretrained_word_embeddings "
                            if helmholtz > 0: 
                                base_parameters += " --synchronous_grammar "
                            
                            # Which primitives set to use.
                            restricted = 'aeioubcdfgsrt' 
                            if restricted in dataset:
                                primitives = f"re2_chars_{restricted} re2_bootstrap_v1_primitives"
                            else:
                                primitives = "re2_chars_None re2_bootstrap_v1_primitives"
                            if use_vowel:
                                primitives += " re2_vowel_consonant_primitives"
                                
                            exp_parameters = f" --taskDataset {dataset} --language_encoder recurrent --languageDataset {dataset}/human --primitives {primitives} --moses_dir ./moses_compiled --smt_phrase_length 1 --smt_pseudoalignments {pseudoalignment}  --language_compression --lc_score {lc_score} --max_compression {max_compression} "
                            if no_consolidation:
                                exp_parameters += "--no-consolidation "
                            
                            exp_command = base_command + base_parameters + exp_parameters
                            command = build_command(exp_command, job, job_name, replication=None)
                            if RUN_HELMHOLTZ_VOWEL_EXPERIMENTS:
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
