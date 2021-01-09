USING_AZURE = False
USING_GCLOUD = False
USING_SINGULARITY = True
NUM_REPLICATIONS = 0
NO_ORIGINAL_REPL = False
HIGH_MEM = False

LANGUAGE_COMPRESSION = True
AZURE_IMAGE = "ec-language-5-24" if not LANGUAGE_COMPRESSION else "ec-language-compression-5-25"
OLD_AZURE_IMAGE = "re2-language-5-20-image-20200520151754 "


# Pretrained checkpoints for using the curriculum.
pretrained_ghelm_checkpoint = "experimentOutputs/clevr/2020-05-22T14-12-31-661550/clevr_aic=1.0_arity=3_ET=14400_it=3_MF=5_no_dsl=F_pc=30.0_RS=10000_RW=F_STM=T_L=1.5_batch=20_K=2_topLL=F_LANG=F.pickle"

def azure_commands(job_name): 
    machine_type = "Standard_D48s_v3"
    azure_launch_command = f"az vm create --name {job_name} --resource-group ec-language-east2 --generate-ssh-keys --data-disk-sizes-gb 128 --image {AZURE_IMAGE}  --size {machine_type} "
    return f"#######\n{azure_launch_command}\n\n###Now run: \n mkdir jobs; "


def gcloud_commands(job_name):
    machine_type = 'm1-ultramem-40' if HIGH_MEM else 'n2-highmem-64'
    gcloud_disk_command = f"gcloud compute --project 'andreas-jacob-8fc0' disks create {job_name} --size '30' --zone 'us-east1-b' --source-snapshot 're2-language-5-14' --type 'pd-standard'"
    gcloud_launch_commmand = f"gcloud beta compute --project=andreas-jacob-8fc0 instances create {job_name} --metadata='startup-script=cd ec' --zone=us-east1-b --machine-type={machine_type} --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=project-service-account@andreas-jacob-8fc0.iam.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --disk=name={job_name.strip()},device-name={job_name.strip()},mode=rw,boot=yes,auto-delete=yes --reservation-affinity=any"
    return f"#######\n{gcloud_disk_command}\n\n{gcloud_launch_commmand}\n\n###Now run: \n "
    
singularity_base_command = "srun --job-name=clevr_language_{} --output=jobs/{} --ntasks=1 --mem-per-cpu=10000 --gres=gpu --cpus-per-task 12 --time=10000:00 --qos=tenenbaum --partition=tenenbaum singularity exec -B /om2  --nv ../dev-container.img "
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

### Generates curriculum experiments for the Helmholtz generative model
RUN_CURR_HELMHOLTZ_GENERATIVE_MODEL = False
num_iterations = 3
task_batch_size = 20
recognition_steps = 10000
enumerationTimeout = 1800
EXPS = [("bootstrap", 3600), ("bootstrap", 7200), ("bootstrap", 14400), ("bootstrap", 21600)]
task_datasets = ("1_zero_hop", "1_one_hop", "1_compare_integer", "1_same_relate", "1_single_or", "2_remove", "2_transform")
primitives = [("bootstrap", ("clevr_bootstrap", "clevr_map_transform")), 
              ("original", ("clevr_original", "clevr_map_transform")),
              ("filter", ("clevr_original", "clevr_map_transform", "clevr_filter")),]
for prim_name, primitive_set in primitives:
    for enumerationTimeout in [1800, 3600, 7200, 14400, 21600]:
        exp = (prim_name, enumerationTimeout)
        job_name = f"clevr_ec_gru_ghelm_compression_et_{enumerationTimeout}_curr_prim_{prim_name}"
        jobs.append(job_name)
        base_command = "python bin/clevr.py "
        
        base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout 0  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size}  --no-cuda --recognitionSteps {recognition_steps} --recognition_0 --recognition_1 examples language --Helmholtz 0.5 --synchronous_grammar --skip_first_test "
        
        prims = " ".join(primitive_set)
        exp_parameters = f" --curriculumDatasets curriculum --taskDatasets  --language_encoder recurrent --primitives {prims} --moses_dir ./moses_compiled --smt_phrase_length 1 --taskReranker sentence_length"
        exp_command = base_command + base_parameters + exp_parameters
        command = build_command(exp_command, job, job_name, replication=None)
        if RUN_CURR_HELMHOLTZ_GENERATIVE_MODEL:
            if (EXPS is None) or (exp in EXPS):
                if not NO_ORIGINAL_REPL: experiment_commands.append(command)
                experiment_commands += build_replications(exp_command, job, job_name)
        job +=1

### Generates curriculum experiments for the pseudoalignments model.
RUN_CURR_HELMHOLTZ_PSEUDOALIGNMENTS = False
num_iterations = 3
task_batch_size = 20
recognition_steps = 10000
enumerationTimeout = 1800
pseudoalignment = 0.1

EXPS = None
task_datasets = ("1_zero_hop", "1_one_hop", "1_compare_integer", "1_same_relate", "1_single_or", "2_remove", "2_transform")
primitives = [("bootstrap", ("clevr_bootstrap", "clevr_map_transform")), 
              ("original", ("clevr_original", "clevr_map_transform")),]
for prim_name, primitive_set in primitives:
    job_name = f"clevr_ec_gru_ghelm_pseudo_compression_et_{enumerationTimeout}_curr_prim_{prim_name}"
    jobs.append(job_name)
    base_command = "python bin/clevr.py "
    
    base_parameters = f" --enumerationTimeout {enumerationTimeout} --testingTimeout 0  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size}  --no-cuda --recognitionSteps {recognition_steps} --recognition_0 --recognition_1 examples language --Helmholtz 0.5 --synchronous_grammar --skip_first_test "
    
    prims = " ".join(primitive_set)
    exp_parameters = f" --curriculumDatasets curriculum --taskDatasets  --language_encoder recurrent --primitives {prims} --moses_dir ./moses_compiled --smt_phrase_length 1 --smt_pseudoalignments {pseudoalignment} "
    exp_command = base_command + base_parameters + exp_parameters
    command = build_command(exp_command, job, job_name, replication=None)
    if RUN_CURR_HELMHOLTZ_PSEUDOALIGNMENTS:
        if (EXPS is None) or (exp in EXPS):
            if not NO_ORIGINAL_REPL: experiment_commands.append(command)
            experiment_commands += build_replications(exp_command, job, job_name)
    job +=1

## Generates experiments using the full generative model.
RUN_HELMHOLTZ_GENERATIVE_MODEL = True
num_iterations = 10
task_batch_size = 20
recognition_steps = 10000
test_every = 3
initial_timeout_iterations = 1
EXPS = [("bootstrap", 7200, 720), ("bootstrap", 1800, 1800)]
task_datasets = ("1_zero_hop", "1_one_hop", "1_compare_integer", "1_same_relate", "1_single_or", "2_remove", "2_transform", "2_localization")
primitives = [("bootstrap", ("clevr_bootstrap", "clevr_map_transform")), 
              ("filter", ("clevr_bootstrap", "clevr_map_transform", "clevr_filter")),]
for prim_name, primitive_set in primitives:
    for initialTimeout in [10000, 1800]:
        for enumerationTimeout in [1800]:
            exp = (prim_name, initialTimeout, enumerationTimeout)
            job_name = f"clevr_ec_gru_ghelm_compression_it_{initialTimeout}_et_{enumerationTimeout}_prim_{prim_name}"
            jobs.append(job_name)
            base_command = "python bin/clevr.py "
            
            base_parameters = f" --initialTimeout {initialTimeout} --initialTimeoutIterations 3 --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size}  --testEvery {test_every}  --no-cuda --recognitionSteps {recognition_steps} --recognition_0 --recognition_1 examples language --Helmholtz 0.5 --synchronous_grammar --skip_first_test "
            
            prims = " ".join(primitive_set)
            tasks = " ".join(task_datasets)
            exp_parameters = f" --curriculumDatasets --taskDatasets {tasks}  --language_encoder recurrent --primitives {prims} --moses_dir ./moses_compiled --smt_phrase_length 1 --lc_score 0 "
            exp_command = base_command + base_parameters + exp_parameters
            command = build_command(exp_command, job, job_name, replication=None)
            if RUN_HELMHOLTZ_GENERATIVE_MODEL:
                if (EXPS is None) or (exp in EXPS):
                    if not NO_ORIGINAL_REPL: experiment_commands.append(command)
                    experiment_commands += build_replications(exp_command, job, job_name)
            job +=1

## Generates experiments using the full generative model.
RUN_HELMHOLTZ_PSEUDOALIGNMENTS = True
num_iterations = 10
task_batch_size = 20
recognition_steps = 10000
test_every = 3
initial_timeout_iterations = 3
pseudoalignment = 0.1
EXPS = [("bootstrap", 1800, 1800)]
task_datasets = ("1_zero_hop", "1_one_hop", "1_compare_integer", "1_same_relate", "1_single_or", "2_remove", "2_transform", "2_localization")
primitives = [("bootstrap", ("clevr_bootstrap", "clevr_map_transform")), 
              ("filter", ("clevr_bootstrap", "clevr_map_transform", "clevr_filter")),]
for prim_name, primitive_set in primitives:
    for initialTimeout in [3600, 7200]:
        for enumerationTimeout in [1800]:
            exp = (prim_name, initialTimeout, enumerationTimeout)
            job_name = f"clevr_ec_gru_ghelm_pseudo_compression_it_{initialTimeout}_et_{enumerationTimeout}_prim_{prim_name}"
            jobs.append(job_name)
            base_command = "python bin/clevr.py "
            
            base_parameters = f" --initialTimeout {initialTimeout} --initialTimeoutIterations 3 --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size}  --testEvery {test_every}  --no-cuda --recognitionSteps {recognition_steps} --recognition_0 --recognition_1 examples language --Helmholtz 0.5 --synchronous_grammar --skip_first_test "
            
            prims = " ".join(primitive_set)
            tasks = " ".join(task_datasets)
            exp_parameters = f" --curriculumDatasets --taskDatasets {tasks}  --language_encoder recurrent --primitives {prims} --moses_dir ./moses_compiled --smt_phrase_length 1 --smt_pseudoalignments {pseudoalignment} "
            exp_command = base_command + base_parameters + exp_parameters
            command = build_command(exp_command, job, job_name, replication=None)
            if RUN_HELMHOLTZ_PSEUDOALIGNMENTS:
                if (EXPS is None) or (exp in EXPS):
                    if not NO_ORIGINAL_REPL: experiment_commands.append(command)
                    experiment_commands += build_replications(exp_command, job, job_name)
            job +=1

## Generates experiments using the full generative model.
RUN_NO_HELMHOLTZ_GENERATIVE_MODEL = True
num_iterations = 10
task_batch_size = 20
recognition_steps = 10000
test_every = 3
initial_timeout_iterations = 3
pseudoalignment = 0.1
EXPS = [("bootstrap", 1800, 1800)]
task_datasets = ("1_zero_hop", "1_one_hop", "1_compare_integer", "1_same_relate", "1_single_or", "2_remove", "2_transform", "2_localization")
primitives = [("bootstrap", ("clevr_bootstrap", "clevr_map_transform")), 
              ("filter", ("clevr_bootstrap", "clevr_map_transform", "clevr_filter")),]
for prim_name, primitive_set in primitives:
    for initialTimeout in [1800, 7200]:
        for enumerationTimeout in [1800]:
            exp = (prim_name, initialTimeout, enumerationTimeout)
            job_name = f"clevr_ec_gru_no_ghelm_compression_it_{initialTimeout}_et_{enumerationTimeout}_prim_{prim_name}"
            jobs.append(job_name)
            base_command = "python bin/clevr.py "
            
            base_parameters = f" --initialTimeout {initialTimeout} --initialTimeoutIterations 3 --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size}  --testEvery {test_every}  --no-cuda --recognitionSteps {recognition_steps} --recognition_0 --recognition_1 examples language --Helmholtz 0 --skip_first_test  "
            
            prims = " ".join(primitive_set)
            tasks = " ".join(task_datasets)
            exp_parameters = f" --curriculumDatasets --taskDatasets {tasks}  --language_encoder recurrent --primitives {prims} --moses_dir ./moses_compiled --smt_phrase_length 1 --smt_pseudoalignments {pseudoalignment} "
            exp_command = base_command + base_parameters + exp_parameters
            command = build_command(exp_command, job, job_name, replication=None)
            if RUN_NO_HELMHOLTZ_GENERATIVE_MODEL:
                if (EXPS is None) or (exp in EXPS):
                    if not NO_ORIGINAL_REPL: experiment_commands.append(command)
                    experiment_commands += build_replications(exp_command, job, job_name)
            job +=1

## Generates experiments using the full generative model.
RUN_EC_BASELINES = True
num_iterations = 10
task_batch_size = 20
recognition_steps = 10000
test_every = 3
initial_timeout_iterations = 3
pseudoalignment = 0.1
EXPS = [("bootstrap", 1800, 1800)]
task_datasets = ("1_zero_hop", "1_one_hop", "1_compare_integer", "1_same_relate", "1_single_or", "2_remove", "2_transform", "2_localization")
primitives = [("bootstrap", ("clevr_bootstrap", "clevr_map_transform")), 
              ("filter", ("clevr_bootstrap", "clevr_map_transform", "clevr_filter")),]
for prim_name, primitive_set in primitives:
    for initialTimeout in [3600, 1800]:
        for enumerationTimeout in [1800]:
            exp = (prim_name, initialTimeout, enumerationTimeout)
            job_name = f"clevr_ec_no_lang_compression_it_{initialTimeout}_et_{enumerationTimeout}_prim_{prim_name}"
            jobs.append(job_name)
            base_command = "python bin/clevr.py "
            
            base_parameters = f" --initialTimeout {initialTimeout} --initialTimeoutIterations 3 --enumerationTimeout {enumerationTimeout} --testingTimeout {enumerationTimeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size}  --testEvery {test_every}  --no-cuda --recognitionSteps {recognition_steps} --recognition_0 examples --Helmholtz 0.5 --skip_first_test "
            
            prims = " ".join(primitive_set)
            tasks = " ".join(task_datasets)
            exp_parameters = f" --curriculumDatasets --taskDatasets {tasks}  --language_encoder recurrent --primitives {prims} --moses_dir ./moses_compiled --smt_phrase_length 1 --smt_pseudoalignments {pseudoalignment} "
            exp_command = base_command + base_parameters + exp_parameters
            command = build_command(exp_command, job, job_name, replication=None)
            if RUN_EC_BASELINES:
                if (EXPS is None) or (exp in EXPS):
                    if not NO_ORIGINAL_REPL: experiment_commands.append(command)
                    experiment_commands += build_replications(exp_command, job, job_name)
            job +=1

## Generates experiments using the full generative model.
RUN_HELMHOLTZ_GENERATIVE_MODEL_LANGUAGE_COMPRESSION  = True
num_iterations = 10
task_batch_size = 20
recognition_steps = 10000
test_every = 3
initial_timeout_iterations = 1
lc_score = 0.2
max_compression = 5
testing_timeout = 720
EXPS = [("bootstrap", 7200, 720), ("bootstrap", 10000, 1800)]
task_datasets = ("1_zero_hop", "1_one_hop", "1_compare_integer", "1_same_relate", "1_single_or", "2_remove", "2_transform", "2_localization")
primitives = [("bootstrap", ("clevr_bootstrap", "clevr_map_transform")), 
              ("filter", ("clevr_bootstrap", "clevr_map_transform", "clevr_filter")),]
for prim_name, primitive_set in primitives:
    for initialTimeout in [7200, 10000]:
        for enumerationTimeout in [1800]:
            exp = (prim_name, initialTimeout, enumerationTimeout)
            job_name = f"clevr_ec_gru_ghelm_compression_it_{initialTimeout}_et_{enumerationTimeout}_prim_{prim_name}"
            jobs.append(job_name)
            base_command = "python bin/clevr.py "
            
            base_parameters = f" --initialTimeout {initialTimeout} --initialTimeoutIterations {initial_timeout_iterations} --enumerationTimeout {enumerationTimeout} --testingTimeout {testing_timeout}  --iterations {num_iterations} --biasOptimal --contextual --taskBatchSize {task_batch_size}  --testEvery {test_every}  --no-cuda --recognitionSteps {recognition_steps} --recognition_0 --recognition_1 examples language --Helmholtz 0.5 --synchronous_grammar --skip_first_test  --taskReranker sentence_length "
            
            prims = " ".join(primitive_set)
            tasks = " ".join(task_datasets)
            exp_parameters = f" --curriculumDatasets --taskDatasets {tasks}  --language_encoder recurrent --primitives {prims} --moses_dir ./moses_compiled --smt_phrase_length 1 --language_compression --lc_score {lc_score} --max_compression {max_compression}  "
            exp_command = base_command + base_parameters + exp_parameters
            command = build_command(exp_command, job, job_name, replication=None)
            if RUN_HELMHOLTZ_GENERATIVE_MODEL_LANGUAGE_COMPRESSION:
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
        print("")
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
