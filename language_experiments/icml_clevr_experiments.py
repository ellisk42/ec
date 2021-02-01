"""
icml_clevr_experiments.py | Author: Catherine Wong.

This generates the command-line output necessary to launch experiments on cloud platforms, such as OpenMind, Azure, and Google Cloud.

Example usage:
python icml_clevr_experiments.py 
    --cloud_platform supercloud
    --number_random_replications
    --experiment_prefix clevr
    --experiment_classes all
    --output_all_commands_at_once
    --generate_resume_command_for_log

Available experiments: 
    no_language_baseline_bootstrap  
    no_joint_generative_baseline_bootstrap
    no_induced_language_no_compression_baseline_bootstrap
    full_language_generative_bootstrap 
    full_language_mutual_exclusivity_bootstrap
    full_language_language_compression_bootstrap
    full_language_both_add_ons_bootstrap
    
    full_language_synth_train_human_test_bootstrap 
    full_language_human_train_human_test_bootstrap
    
    best_dsl_no_language_pure_enumeration_bootstrap
    best_dsl_full_language_pure_enumeration_and_neural_search_bootstrap
All experiments must be manually added to an experiment registry in register_all_experiment. The procedure for adding in an experiment is: add a human readable string tag; add a 'build_experiment' function that constructs the experiment-specific parameters, and then add that to the registry.
"""
GENERATE_ALL_FLAG = 'all' # Generate all experiment tasks.
DEFAULT_CLEVR_DOMAIN_NAME_PREFIX = 'clevr'
DEFAULT_LOG_DIRECTORY = f"../ec_language_logs/{DEFAULT_CLEVR_DOMAIN_NAME_PREFIX}"
DEFAULT_PYTHON_MAIN_COMMAND = f"python bin/clevr.py "

# Default parameters for running on OpenMind
OM_FLAG = 'om'
OM_SCP_COMMAND = "zyzzyva@openmind7.mit.edu" # Specific to Catherine Wong
DEFAULT_SLURM_CPUS_PER_TASK = 24
DEFAULT_SLURM_TIME_PER_TASK = 10000

# Default parameters for running on Supercloud
SUPERCLOUD_FLAG = 'supercloud'
SUPERCLOUD_SCP_COMMAND = "zyzzyva@txe1-login.mit.edu" # Specific to Catherine Wong

# Default parameters for running on Azure.
AZ_SCP_COMMAND = "zyzzyva@MUST FILL IN"
AZURE_FLAG = 'azure'
AZURE_RESOURCE_GROUP = "resource-group"
AZURE_LOCATION = "location"
AZURE_IMAGE = "image"
AZURE_DEFAULT_MACHINE_TYPE = "Standard_E48s_v3"
AZURE_DEFAULT_RESOURCES_0 = { 
    AZURE_RESOURCE_GROUP : "zyzzyva-clevr-0",
    AZURE_LOCATION : "eastus",
    AZURE_IMAGE : "myGallery/zyzzyva-clevr-0-base-image/0.0.1"
}
AZURE_DEFAULT_RESOURCES_1 = { 
    AZURE_RESOURCE_GROUP : "zyzzyva-clevr-1",
    AZURE_LOCATION : "eastus2",
    AZURE_IMAGE : "myGallery/zyzzyva-clevr-1-base-image/0.0.1"
}
AZURE_RESOURCES = {
    AZURE_DEFAULT_RESOURCES_0[AZURE_RESOURCE_GROUP] : AZURE_DEFAULT_RESOURCES_0,
    AZURE_DEFAULT_RESOURCES_1[AZURE_RESOURCE_GROUP] : AZURE_DEFAULT_RESOURCES_1
}
# Default parameters for each experiment.
DEFAULT_TASK_BATCH_SIZE = 40
DEFAULT_RECOGNITION_STEPS = 10000
DEFAULT_TEST_EVERY = 1
DEFAULT_ITERATIONS = 10
DEFAULT_ENUMERATION_TIMEOUT = 1000
DEFAULT_MEM_PER_ENUMERATION_THREAD = 5000000000 # 5 GB
DEFAULT_BOOTSTRAP_PRIMITIVES_STRING = 'clevr_bootstrap clevr_map_transform' # Primitives for the bootstrap primitives experiments.
DEFAULT_PSEUDOALIGNMENTS_WEIGHT = 0.05
DEFAULT_LANGUAGE_COMPRESSION_WEIGHT = 0.05
DEFAULT_LANGUAGE_COMPRESSION_MAX_COMPRESSION = 5 

DEFAULT_CLEVR_HUMAN_LANGUAGE_DATASET_DIR = f'data/{DEFAULT_CLEVR_DOMAIN_NAME_PREFIX}/language_human'

# Tags for experiment classes.
EXPERIMENTS_REGISTRY = dict()
EXPERIMENT_TAG_NO_LANGUAGE_BASELINE_BOOTSTRAP = 'no_language_baseline_bootstrap'
EXPERIMENT_TAG_NO_JOINT_GENERATIVE_BASELINE_BOOTSTRAP = 'no_joint_generative_baseline_bootstrap'
EXPERIMENT_TAG_NO_INDUCED_LANGUAGE_NO_COMPRESION_BASELINE_BOOTSTRAP = 'no_induced_language_no_compression_baseline_bootstrap'
EXPERIMENT_TAG_FULL_LANGUAGE_GENERATIVE_BOOTSTRAP = 'full_language_generative_bootstrap'
EXPERIMENT_TAG_FULL_LANGUAGE_MUTUAL_EXCLUSIVITY_BOOTSTRAP = 'full_language_mutual_exclusivity_bootstrap'
EXPERIMENT_TAG_FULL_LANGUAGE_LANGUAGE_COMPRESSION_BOOTSTRAP = 'full_language_language_compression_bootstrap'
EXPERIMENT_TAG_FULL_LANGUAGE_BOTH_ADD_ONS_BOOTSTRAP = 'full_language_both_add_ons_bootstrap'
EXPERIMENT_TAG_NO_INDUCED_LANGUAGE_NO_COMPRESION_BASELINE_BOOTSTRAP = 'no_induced_language_no_compression_baseline_bootstrap'

EXPERIMENT_TAG_FULL_LANGUAGE_SYNTH_TRAIN_HUMAN_TEST = 'full_language_synth_train_human_test_bootstrap'
EXPERIMENT_TAG_FULL_LANGUAGE_HUMAN_TRAIN_HUMAN_TEST = 'full_language_human_train_human_test_bootstrap'

EXPERIMENT_TAG_BEST_DSL_NO_LANGUAGE_PURE_ENUMERATION = 'best_dsl_no_language_pure_enumeration_bootstrap'
EXPERIMENT_TAG_BEST_DSL_FULL_LANGUAGE_PURE_ENUMERATION_AND_NEURAL_SEARCH = 'best_dsl_full_language_pure_enumeration_and_neural_search_bootstrap'

GLOBAL_EXPERIMENTS_ARGUMENTS = dict() # Tracks globally set arguments.
USER_INPUT_DEFAULT_PARAMETERS = dict() # Tracks the latest user input parameter so we can offer it as a default.
NUM_CPUS_TAG = 'CPUs'
import os
import sys
import subprocess
import argparse
import datetime
from collections import defaultdict
parser = argparse.ArgumentParser()
parser.add_argument('--cloud_platform',
                    default=OM_FLAG,
                    help='Which cloud platform you are attempting to generate commands for.')
parser.add_argument('--number_random_replications',
                    default=1,
                    type=int,
                    help='The total number of replications to run for a given experiment type.')
parser.add_argument('--experiment_prefix',
                    default=DEFAULT_CLEVR_DOMAIN_NAME_PREFIX,
                    help='The experimental prefix that will be appended to the experiment.')
parser.add_argument('--experiment_classes',
                    nargs='*',
                    default=[],
                    help="Which experiments to run. 'all' for all of the ones currently in the registry.")
parser.add_argument('--experiment_log_directory',
                    default=DEFAULT_LOG_DIRECTORY,
                    help="The logging output directory to which we will output the logging information.")
parser.add_argument('--output_all_commands_at_once',
                    action='store_true',
                    help="If true, we print all commands at once, rather than iteratively by experiment type.")
parser.add_argument('--generate_resume_command_for_log',
                    help="Generates a resume command for a log. Looks for the log in cached logs, otherwise SCPs it.")

def optionally_generate_resume_command_for_log(args):
    """ Extracts checkpoints to resume from if provided. These can be used to find a matching experiment from which to resume.
    Returns: {basename to resume : checkpoint}
    """
    if not args.generate_resume_command_for_log: return dict()
    remote_log_file_to_resume = args.generate_resume_command_for_log
    print(f"Going to generate a resume command for logfile: {remote_log_file_to_resume}")
    local_logfile_path = get_cached_log_if_exists(remote_log_file_to_resume, args)
    if not local_logfile_path:
        local_logfile_path = get_remote_logfile_via_scp(remote_log_file_to_resume, args)
    checkpoint_to_resume = extract_checkpoint_from_logfile(local_logfile_path)
    experiment_basename = get_experiment_basename_from_logfile_path(local_logfile_path)
    return {experiment_basename : checkpoint_to_resume}
    if not(input("Continue on to experiments? Default: will exit")):
        sys.exit(0)

def get_experiment_basename_from_logfile_path(logfile_path):
    """Re-extracts an experiment basename from the logfile path based on the delimiter. Expects {prefix}-{BASENAME}-special values"""
    logfile_basename = os.path.basename(logfile_path)
    experiment_basename = logfile_basename.split("-")[-2]
    return experiment_basename

def get_cached_log_if_exists(remote_logfile_path, args):
    """Returns the filepath to a cached local logfile if it exists and the user agrees that they want it."""
    logfile_basename = os.path.basename(remote_logfile_path)
    local_logfile_path = os.path.join(args.experiment_log_directory, logfile_basename)
    if os.path.exists(local_logfile_path):
        if not input(f"Use a cached local logfile?: {local_logfile_path} (Default: yes)"):
            return local_logfile_path
    return None

def get_remote_logfile_via_scp(remote_logfile_path, args):
    """SCPs the remote logfile and returns the filepath to a cached local logfile."""
    logfile_basename = os.path.basename(remote_logfile_path)
    local_logfile_path = os.path.join(args.experiment_log_directory, logfile_basename)
    if args.cloud_platform == OM_FLAG:
        scp_command = f"scp {OM_SCP_COMMAND}:{remote_logfile_path} {local_logfile_path}"
        print(f"Logfile not found locally, SCPing from: {scp_command }")
        subprocess.check_output(scp_command, shell=True)
        return local_logfile_path
    elif args.cloud_platform == SUPERCLOUD_FLAG:
        scp_command = f"scp {SUPERCLOUD_SCP_COMMAND}:{remote_logfile_path} {local_logfile_path}"
        print(f"Logfile not found locally, SCPing from: {scp_command }")
        subprocess.check_output(scp_command, shell=True)
        return local_logfile_path
    elif args.cloud_platform == AZURE_FLAG:
        azure_location = input("What is the Azure SCP path?")
        scp_command = f"scp {azure_location}:{remote_logfile_path} {local_logfile_path}"
        print(f"Logfile not found locally, SCPing from: {scp_command }")
        subprocess.check_output(scp_command, shell=True)
        return local_logfile_path
    else:
        print(f"Unknown cloud platform for SCP: {args.cloud_platform}")
        sys.exit(0)

def extract_checkpoint_from_logfile(local_logfile_path):
    """Extracts potential checkpoints from the logfile and gets the one the user wants to resume."""
    CHECKPOINT_STRING = 'Exported checkpoint to '
    with open(local_logfile_path, 'r') as f:
        lines = f.readlines()
    # Checkpoint lines are in a canonical form: Exported checkpoint to ___ path.
    checkpoints = [line.split(CHECKPOINT_STRING)[1].strip() for line in lines if CHECKPOINT_STRING in line] 
    if len(checkpoints) == 0:
        print("No checkpoints found for resumption. Exiting.")
        sys.exit(0)
    last_checkpoint = checkpoints[-1]
    iteration_to_resume = input(f"Iteration to resume (int) Default: {last_checkpoint}")
    if not iteration_to_resume: 
        checkpoint_to_resume = last_checkpoint
    else:
        # Look for the checkpoint of that iteration.
        checkpoint_iteration_tag = f'_it={iteration_to_resume}_'
        checkpoints_to_resume = [checkpoint for checkpoint in checkpoints if checkpoint_iteration_tag in checkpoint] 
        if len(checkpoints_to_resume) != 1:
            print("Did not find a matching checkpoint. Exiting")
            sys.exit(0)
        checkpoint_to_resume = checkpoints_to_resume[0]
    print(f"Found checkpoint {checkpoint_to_resume} to resume")
    return checkpoint_to_resume

def generate_timestamped_record_for_csv_logs(args, experiment_name, all_final_commands, experiment_information_dict):
    """Generates a record of these experiments for the CSV logs."""
    experiment_tags = get_input_or_default("Comma-separated experiment-level tags?", "")
    # Builds a list based on the current spreadsheet layout.
    csv_log_entries_for_experiment = []
    for replication_idx, replication_full_command in enumerate(all_final_commands):
        if args.cloud_platform == AZURE_FLAG:
            replication_full_command = replication_full_command.replace("\n", " COMMAND ")
        logfile_path, logfile_command = get_logfile_path_and_scp_command(args, replication_full_command)
        csv_log_entry = [ #The tags in each tuple are only for our own readability.
            ("completed_success_fail", ""),
            ("last_monitored_iteration", str(0)),
            ("last_monitored_test_accuracy", "--/-- --"),
            ("last_monitored_date",  get_current_date_formatted()),
            ("launch_date",  get_current_date_formatted()),
            ("experiment_class",  experiment_information_dict['experiment_basename']),
            ("experiment_name", experiment_name),
            ("replication_idx", f"{replication_idx} / {len(all_final_commands)}"),
            ("experiment_parameters", experiment_information_dict['interactive_parameters']),
            ("tags", experiment_tags),
            ("compute_class", args.cloud_platform),
            ("logfile_location", logfile_path),
            ("scp_logfile_command", logfile_command),
            ("original_launch_command", replication_full_command)
            
        ]
        csv_log_entries_for_experiment.append(csv_log_entry)
    # Prints out the entries as CSV lines
    for csv_log_entry in csv_log_entries_for_experiment:
        print(",".join([value for (key, value) in csv_log_entry]))
        print("\n")
    print("\n")

def get_current_date_formatted():
    """Utility to get the current date, formatted by Month-Date-Year-Hour"""
    current_time = datetime.datetime.now()
    return current_time.strftime("%m-%d-%y %I:00%p")

def get_logfile_path_and_scp_command(args, full_command):
    """Utility to get the logfile path and the SCP command to get the file, based on the launch platform."""
    if args.cloud_platform == OM_FLAG:
        logfile_path = full_command.split('--output=')[1].split(" ")[0]
        full_logfile_path = logfile_path.replace('..', '/om2/user/zyzzyva')
        scp_command = f"scp {OM_SCP_COMMAND}:{full_logfile_path} {logfile_path} "
        return full_logfile_path, scp_command
    elif args.cloud_platform == SUPERCLOUD_FLAG:
        logfile_path = full_command.split('--output=')[1].split(" ")[0]
        full_logfile_path = logfile_path.replace('..', '/home/gridsan/zyzzyva/mit')
        scp_command = f"scp {SUPERCLOUD_SCP_COMMAND}:{full_logfile_path} {logfile_path} "
        return full_logfile_path, scp_command
    elif args.cloud_platform == AZURE_FLAG:
        logfile_path = full_command.split(" > ")[-1].split()[0]
        full_logfile_path = logfile_path.replace('..', '~')
        scp_command = f"scp {AZ_SCP_COMMAND}:{full_logfile_path} {logfile_path} "
        return full_logfile_path, scp_command
    else:
        print(f"Unknown cloud platform: {args.cloud_platform}")
        sys.exit(0)
    
def output_launch_commands_and_log_lines(cloud_launcher_command, experiment_commands, args):
    """Outputs the launch commands for a set of experiments, and a comma separated line that can be logged in an experiment spreadsheet.
    """
    for experiment_name in experiment_commands:
        print(f"Outputting experiment commands for: {experiment_name}\n")
        all_final_commands = []
        
        experiment_information_dict = experiment_commands[experiment_name]
        replication_commands = experiment_information_dict['replication_commands']
        resume_command = experiment_information_dict['resume_command']
        is_random = len(replication_commands) > 1
        for replication_idx, replication_command in enumerate(replication_commands):
            final_command = build_final_command_from_launcher_and_experiment(cloud_launcher_command, experiment_name, replication_command, replication_idx, resume_command, is_random=is_random)
            all_final_commands.append(final_command)
            print(final_command)
            print("\n")
        generate_timestamped_record_for_csv_logs(args, experiment_name, all_final_commands, experiment_information_dict)
        if not args.output_all_commands_at_once:
            input("....hit enter for next experiments\n")
            
def build_final_command_from_launcher_and_experiment(cloud_launcher_command, experiment_name, replication_command, replication_idx, resume_command, is_random=False):
    """Builds the final launch command from a given experimental command and its cloud launcher."""
    rand_tag = "_rand" if is_random else ""
    if args.cloud_platform == OM_FLAG or args.cloud_platform == SUPERCLOUD_FLAG:
        # This builds the output file name, which is of the form {experiment_name}_replication_idx.
        formatted_launch_command = cloud_launcher_command.format(experiment_name, str(replication_idx)+rand_tag, experiment_name, str(replication_idx)+rand_tag, is_random)
        return formatted_launch_command + replication_command + resume_command + "' &"
    if args.cloud_platform == AZURE_FLAG:
        job_name = experiment_name+ str(replication_idx)+rand_tag
        job_name = job_name.replace("_", "-")
        # Truncate for Azure
        job_name = job_name[:30] + job_name[-30:]
        formatted_launch_command = cloud_launcher_command.format(job_name) + replication_command + resume_command + f" > ../ec_language_logs/{experiment_name+ str(replication_idx)+rand_tag} 2>&1 &"
        return formatted_launch_command
    else:
        print(f"Unknown cloud platform: {args.cloud_platform}")
        sys.exit(0)
    
def build_cloud_launcher_command(args):
    """Builds the cloud launcher command for a given cloud platform, with additional prompts if needed.
    """
    if args.cloud_platform == OM_FLAG:
        return build_om_launcher_command(args)
    elif args.cloud_platform == SUPERCLOUD_FLAG:
        return build_supercloud_launcher_command(args)
    elif args.cloud_platform == AZURE_FLAG:
        return build_azure_launcher_command(args)
    else:
        print(f"Unknown cloud platform: {args.cloud_platform}")
        sys.exit(0)

def build_default_slurm_command(args):
    """Builds a launcher command for clusters that use SLURM as their batch job system."""
    print("Running on a slurm machine (OM or Supercloud). Please input the following parameters:")
    number_cpus_per_task = get_input_or_default("Number of CPUS per task?", DEFAULT_SLURM_CPUS_PER_TASK)
    # This requires limitation on the program side, so we set it in the global dictionary.
    GLOBAL_EXPERIMENTS_ARGUMENTS[NUM_CPUS_TAG] = number_cpus_per_task
    slurm_base_command = f"sbatch --job-name="+args.experiment_prefix+"-language-{}_{} --output="+args.experiment_log_directory+"/" +args.experiment_prefix+"-{}_{} --ntasks=1 --mem=MaxMemPerNode --cpus-per-task "+str(number_cpus_per_task)
    print("\n")
    return slurm_base_command
    
def build_om_launcher_command(args):
    """Builds the launcher command for running on OpenMind. 
    Returns a string command that can be run """
    om_base_command = build_default_slurm_command(args) + " --time="+ str(DEFAULT_SLURM_TIME_PER_TASK) + ":00 " + " --qos=tenenbaum --partition=tenenbaum  --wrap='singularity exec -B /om2  --nv ../dev-container.img "
    print("\n")
    return om_base_command

def build_supercloud_launcher_command(args):
    """Builds the launcher command for running on OpenMind. 
    Returns a string command that can be run """
    supercloud_base_command = build_default_slurm_command(args) + " --exclusive  --wrap='singularity exec ../dev-container.img  "
    print("\n")
    return supercloud_base_command

def build_azure_launcher_command(args):
    """Builds the launcher command for running on Azure. Azure requires an additional login and computer creation step."""
    # First we will need to provision a machine.
    azure_resource_data = get_input_or_default("Which Azure resource group do you want? ",  AZURE_DEFAULT_RESOURCES_0[AZURE_RESOURCE_GROUP])
    azure_resource_data = AZURE_RESOURCES[azure_resource_data]
    group = azure_resource_data[AZURE_RESOURCE_GROUP]
    image = azure_resource_data[AZURE_IMAGE]
    location = azure_resource_data[AZURE_LOCATION]
    azure_machine = get_input_or_default("Azure machine type? ",  AZURE_DEFAULT_MACHINE_TYPE)
    azure_launch_command = "az vm create --name az-{}" + f" --resource-group {group} --generate-ssh-keys --data-disk-sizes-gb 128 --image {image} --size {azure_machine} --location {location} \n\n singularity exec ../dev-container.img  "
    return azure_launch_command
    

def build_experiment_commands(args, experiment_to_resume_checkpoint):
    """Given a set of experiment tags to run, builds the appropriate commands from the registry, including their replications.
    
    experiment_to_resume_checkpoint is of the form {experiment_name_to_resume : checkpoints to resume}
    
    Returns: dict {
        full_experiment_name : {
            'replication_commands' : all_replication_commands,
            'experiment_basename' : basename,
            'interactive_parameters' : interactive_experiment_parameters,
            'resume_command' : a command if we are resuming, else empty string
        }
    }
    """
    experiment_classes_to_resume = list(experiment_to_resume_checkpoint.keys())
    experiment_classes = args.experiment_classes + experiment_classes_to_resume
    if args.experiment_classes == [GENERATE_ALL_FLAG]:
        experiment_classes = EXPERIMENTS_REGISTRY.keys()
    experiment_commands_dict = defaultdict(list)
    for experiment_class in experiment_classes:
        if experiment_class not in EXPERIMENTS_REGISTRY:
            print(f"Not found in the experiments registry: {experiment_class}")
            sys.exit(0)
        if experiment_class in experiment_to_resume_checkpoint:
            print("Generating a resume command.")

        experiment_command_builder_fn = EXPERIMENTS_REGISTRY[experiment_class]
        experiment_name, experiment_information_dict = experiment_command_builder_fn(experiment_class, args, experiment_to_resume_checkpoint)
        experiment_name = add_resume_commands(experiment_class, experiment_name, experiment_information_dict, experiment_to_resume_checkpoint)
        experiment_commands_dict[experiment_name] = experiment_information_dict
    return experiment_commands_dict

def add_resume_commands(experiment_class, experiment_name, experiment_information_dict, experiment_to_resume_checkpoint):
    """
    Adds the resume command to the experiment information_dict if we have found an appropriate checkpoint.
    Updates the experiment name to contain the starting iteration.
    Mutates: experiment_information_dict
    """
    starting_iteration = 0
    experiment_information_dict['resume_command'] = " "
    if experiment_class in experiment_to_resume_checkpoint:
        checkpoint_to_resume = experiment_to_resume_checkpoint[experiment_class]
        experiment_information_dict['resume_command'] = f" --resume {checkpoint_to_resume} "
        starting_iteration = get_iteration_from_resume_checkpoint(checkpoint_to_resume)
    experiment_name = f"{experiment_name}_start_{starting_iteration}"
    return experiment_name

def get_iteration_from_resume_checkpoint(checkpoint_to_resume):
    """Extracts the integer string of the checkpoint, which has an '_it=ITERATION_' substring."""
    iteration_string = checkpoint_to_resume.split("_it=")[1].split("_")[0]
    return int(iteration_string)
        
def build_replication_commands(experiment_command, args):
    """Takes a basic experiment class command and builds a set of replication commands for it.
    This prompts the user if we want to use a sentence_ordered curriculum instead.
     Returns : [array of experiment_replication_commands]"""
    use_sentence_ordered_curriculum = get_input_or_default("Use a sentence length curriculum? ", "yes")
    
    if use_sentence_ordered_curriculum == "yes":
        return [
            experiment_command + f' --taskReranker sentence_length --seed {replication_idx} '
            for replication_idx in range(1, args.number_random_replications + 1)
        ]
    else:
        return [
            experiment_command + f' --taskReranker randomShuffle --seed {replication_idx} '
            for replication_idx in range(1, args.number_random_replications + 1)
        ]

def get_input_or_default(input_string, default_value):
    """Utility method for a common pattern of asking user for input or getting defaults. We then store the default for next time."""
    if input_string in USER_INPUT_DEFAULT_PARAMETERS:
        previous_user_input_value = USER_INPUT_DEFAULT_PARAMETERS[input_string]
        user_input_value = input(f"{input_string} (Use previous value {previous_user_input_value}? Original default was {default_value})")
        value_to_return = user_input_value or previous_user_input_value
    else:
        user_input_value = input(f"{input_string} (Default: {default_value})")
        value_to_return = user_input_value or default_value
    if user_input_value:
        USER_INPUT_DEFAULT_PARAMETERS[input_string] = value_to_return
    return value_to_return
    
def get_interactive_experiment_parameters():
    """Prompts the user for interactive experiment parameters, which vary for most experiments. 
    Returns a set of experiment parameters as a command.
    A string tag that distinguishes this particular experiment"""
    task_datasets = get_input_or_default("Tasks to test on?", GENERATE_ALL_FLAG)
    task_batch_size = get_input_or_default("Task batch size?", DEFAULT_TASK_BATCH_SIZE)
    number_of_iterations = get_input_or_default("Number of iterations?", DEFAULT_ITERATIONS)
    test_every = get_input_or_default("Test on every N iterations?", DEFAULT_TEST_EVERY)
    enumeration_timeout = get_input_or_default("Enumeration timeout per task? Same as testing timeout.", DEFAULT_ENUMERATION_TIMEOUT)
    recognition_steps = get_input_or_default("Total recognition steps? ", DEFAULT_RECOGNITION_STEPS)
    rep_tag = get_input_or_default("Add any replication tag to avoid namespace collision? ", "")
    
    interactive_experiment_parameters = f"--enumerationTimeout {enumeration_timeout} --testingTimeout {enumeration_timeout} --iterations {number_of_iterations} --taskBatchSize {task_batch_size} --testEvery {test_every} --taskDatasets {task_datasets} --recognitionSteps {recognition_steps} "
    
    interactive_experiment_tag = f"et_{enumeration_timeout}_it_{number_of_iterations}_batch_{task_batch_size}"
    if len(rep_tag) > 0:
        interactive_experiment_tag += f"_{rep_tag}"
    return interactive_experiment_parameters, interactive_experiment_tag  

def get_shared_experiment_parameters():
    """Gets parameters that are shared across all experiments. Returns a set of experiment parameters as a command."""
    max_mem_per_enumeration_thread = get_input_or_default("Maximum memory per enumeration thread?", DEFAULT_MEM_PER_ENUMERATION_THREAD)
    
    # Add any global parameters.
    global_parameters_command = " ".join([f"--{global_param} {global_param_value} " for (global_param, global_param_value) in GLOBAL_EXPERIMENTS_ARGUMENTS.items()])
    
    return f"--biasOptimal --contextual --no-cuda --moses_dir ./moses_compiled --smt_phrase_length 1 --language_encoder recurrent --max_mem_per_enumeration_thread {max_mem_per_enumeration_thread} " + global_parameters_command

def get_shared_full_language_experiment_parameters(primitives_string):
    """Gets experiment parameters that are shared across all the full language experiments. Takes a string indicating which primitives to use.
    Returns a set of experiment parameters as a command."""
    return f"--recognition_0 --recognition_1 examples language --Helmholtz 0.5 --primitives {primitives_string}  --synchronous_grammar "

def build_experiment_command_information(basename, args, experiment_parameters_fn):
    """Builds an experiment command by querying for interactive parameters, then running a query for any experiment parameters. 
        Returns: 
            full_experiment_name, experiment_information_dict = {
                'replication_commands' : all_replication_commands,
                'experiment_basename' : basename,
                'interactive_parameters' : interactive_experiment_parameters,
            }
    """
    print(f"Building an experiment for class: {basename}.")
    interactive_experiment_parameters, interactive_experiment_tag  = get_interactive_experiment_parameters() 
    shared_experiment_parameters = get_shared_experiment_parameters()
    experiment_class_parameters = experiment_parameters_fn()
    experiment_command = DEFAULT_PYTHON_MAIN_COMMAND + interactive_experiment_parameters + shared_experiment_parameters + experiment_class_parameters
    
    all_replication_commands = build_replication_commands(experiment_command, args)
    experiment_name = f"{basename}-{interactive_experiment_tag}"
    experiment_information_dict = {
        'replication_commands' : all_replication_commands,
        'experiment_basename' : basename,
        'interactive_parameters' : interactive_experiment_parameters,
    }
    return experiment_name, experiment_information_dict

def build_experiment_baseline_bootstrap_primitives(basename, args, experiment_to_resume_checkpoint):
    """Builds the baseline experiments: these run DreamCoder without any language in the loop. Uses bootstrap primitives.
    """
    def experiment_parameters_fn():
        return  " --recognition_0 examples --Helmholtz 0.5 --primitives clevr_bootstrap clevr_map_transform "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

def build_experiment_no_induced_language_no_compression_baseline_bootstrap_primitives(basename, args, experiment_to_resume_checkpoint):
    """Builds a baseline experiment that only trains directly on the language annotations, with no compression.
    """
    def experiment_parameters_fn():
        return  " --recognition_0 --recognition_1 examples language --Helmholtz 0 --primitives clevr_bootstrap clevr_map_transform  --no-consolidation "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

def build_experiment_no_joint_generative_bootstrap_primitives(basename, args, experiment_to_resume_checkpoint):
    """ Builds a baseline experiment that only trains directly on the language annotations, but with compression.
    """
    def experiment_parameters_fn():
        return  " --recognition_0 --recognition_1 examples language --Helmholtz 0 --primitives clevr_bootstrap clevr_map_transform "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

def build_experiment_full_language_generative_bootstrap_primitives(basename, args, experiment_to_resume_checkpoint):
    """Builds the basic full language experiment: this uses language, but not any pseudoalignments or language-guided compression. Uses bootstrap primitives.
    Returns: 
        full_experiment_name, [array of experiment replication commands]
    """
    def experiment_parameters_fn():      
        return get_shared_full_language_experiment_parameters(primitives_string=DEFAULT_BOOTSTRAP_PRIMITIVES_STRING) + "--lc_score 0 "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

def build_experiment_full_language_mutual_exclusivity_bootstrap_primitives(basename, args, experiment_to_resume_checkpoint):
    """Builds the full language experiment with the 'mutual exclusivity' prior added on top. Uses bootstrap primitives.
    """
    def experiment_parameters_fn():
        pseudoalignments_weight = get_input_or_default("Pseudoalignments weight for mutual exclusivity?", DEFAULT_PSEUDOALIGNMENTS_WEIGHT)
        
        return get_shared_full_language_experiment_parameters(primitives_string=DEFAULT_BOOTSTRAP_PRIMITIVES_STRING) + f"--lc_score 0 --smt_pseudoalignments {pseudoalignments_weight} "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

def build_experiment_full_language_language_compression_bootstrap_primitives(basename, args, experiment_to_resume_checkpoint):
    """Builds the full language experiment with language-guided compression added on top. Uses bootstrap primitives.
    """
    def experiment_parameters_fn():
        language_compression_score = get_input_or_default("Language compression score?", DEFAULT_LANGUAGE_COMPRESSION_WEIGHT)
        max_compression = get_input_or_default("Maximum number of abstractions to add to compression", DEFAULT_LANGUAGE_COMPRESSION_MAX_COMPRESSION)
        return  get_shared_full_language_experiment_parameters(primitives_string=DEFAULT_BOOTSTRAP_PRIMITIVES_STRING) + f"--lc_score {language_compression_score} --max_compression {DEFAULT_LANGUAGE_COMPRESSION_MAX_COMPRESSION} "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

def build_experiment_full_language_both_add_ons_bootstrap_primitives(basename, args, experiment_to_resume_checkpoint):
    """Builds the full language experiment with both mutual exclusivity and language-guided compression added on top. Uses bootstrap primitives.
    """
    def experiment_parameters_fn():
        pseudoalignments_weight = get_input_or_default("Pseudoalignments weight for mutual exclusivity?", DEFAULT_PSEUDOALIGNMENTS_WEIGHT)
        language_compression_score = get_input_or_default("Language compression score?", DEFAULT_LANGUAGE_COMPRESSION_WEIGHT)
        max_compression = get_input_or_default("Maximum number of abstractions to add to compression", DEFAULT_LANGUAGE_COMPRESSION_MAX_COMPRESSION)
        return  get_shared_full_language_experiment_parameters(primitives_string=DEFAULT_BOOTSTRAP_PRIMITIVES_STRING) + f" --smt_pseudoalignments {pseudoalignments_weight} --lc_score {language_compression_score} --max_compression {DEFAULT_LANGUAGE_COMPRESSION_MAX_COMPRESSION} "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)


def build_experiment_full_language_human_train_human_test_bootstrap(basename, args, experiment_to_resume_checkpoint):
    """Builds an experiment that trains and tests the full language model on the human data. This runs the best model so far, which we assume is the baseline full language model. 
    """
    def experiment_parameters_fn():      
        language_dataset_dir = get_input_or_default("Human language dataset directory?", DEFAULT_CLEVR_HUMAN_LANGUAGE_DATASET_DIR)
        return get_shared_full_language_experiment_parameters(primitives_string=DEFAULT_BOOTSTRAP_PRIMITIVES_STRING) + f"--lc_score 0 --languageDatasetDir {language_dataset_dir}  "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

def get_test_only_experiment_shared_parameters_with_checkpoint(basename, args, experiment_to_resume_checkpoint, primitives_string):
    """Experiments that are test only, requiring a pre-trained checkpoint."""
    print("Running an experiment that will only run tests using a pre-trained checkpoint.\n")
    assert len(experiment_to_resume_checkpoint) == 1
    original_experiment_basename = list(experiment_to_resume_checkpoint.keys())[0]
    print(f"Resuming a checkpoint from {original_experiment_basename} for test-only: {basename}")
    
    checkpoint_to_resume = experiment_to_resume_checkpoint.pop(original_experiment_basename)
    experiment_to_resume_checkpoint[basename] = checkpoint_to_resume
    starting_iteration = int(get_iteration_from_resume_checkpoint(checkpoint_to_resume))
    print("When prompted for iterations now, you should type in the NUMBER OF ITERATIONS + 1 of the checkpoint. You will be prompted again later, but the default will be set properly.")
    number_of_iterations = get_input_or_default("Number of iterations?", starting_iteration + 1)
    
    return f"--recognition_0 examples --Helmholtz 0.5 --primitives {primitives_string}  --synchronous_grammar  --no-dsl --no-consolidation --test_dsl_only "

def build_experiment_full_language_synth_train_human_test_bootstrap(basename, args, experiment_to_resume_checkpoint):
    """Builds an experiment that tests on a human language dataset. This requires a checkpoint.
    """
    def experiment_parameters_fn():      
        test_only_parameters = get_test_only_experiment_shared_parameters_with_checkpoint(basename, args, experiment_to_resume_checkpoint, primitives_string=DEFAULT_BOOTSTRAP_PRIMITIVES_STRING)
        # We actually do want to test with a DSL.
        language_dataset_dir = get_input_or_default("Human language dataset directory?", DEFAULT_CLEVR_HUMAN_LANGUAGE_DATASET_DIR)
        test_only_parameters = test_only_parameters.replace("--test_dsl_only", "--skip_first_test")
        return test_only_parameters + f"--languageDatasetDir {language_dataset_dir} --test_only_after_recognition  "
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

def build_experiment_best_dsl_no_language_pure_enumeration_bootstrap(basename, args, experiment_to_resume_checkpoint):
    """Builds an experiment that tests the best DSL learned without language on pure enumeration."""
    def experiment_parameters_fn():      
        return get_test_only_experiment_shared_parameters_with_checkpoint(basename, args, experiment_to_resume_checkpoint, primitives_string=DEFAULT_BOOTSTRAP_PRIMITIVES_STRING)
    return build_experiment_command_information(basename, args, experiment_parameters_fn)

def build_experiment_best_dsl_full_language_pure_enumeration_and_neural_search_bootstrap(basename, args, experiment_to_resume_checkpoint):
    """Builds an experiment that tests the best DSL learned with language on pure enumeration."""
    def experiment_parameters_fn():      
        return get_test_only_experiment_shared_parameters_with_checkpoint(basename, args, experiment_to_resume_checkpoint, primitives_string=DEFAULT_BOOTSTRAP_PRIMITIVES_STRING) + " --test_only_after_recognition "
    return build_experiment_command_information(basename, args, experiment_parameters_fn) 
    
def register_all_experiments():
    """Adds functions for a given experiment type to a global registry.
    Mutates: EXPERIMENTS_REGISTRY
    """
    print("Registering experiments for CLEVR...")
    EXPERIMENTS_REGISTRY[EXPERIMENT_TAG_NO_LANGUAGE_BASELINE_BOOTSTRAP] = build_experiment_baseline_bootstrap_primitives # Baseline experiment.
    EXPERIMENTS_REGISTRY[EXPERIMENT_TAG_NO_JOINT_GENERATIVE_BASELINE_BOOTSTRAP] = build_experiment_no_joint_generative_bootstrap_primitives # No Helmholtz baseline.
    EXPERIMENTS_REGISTRY[EXPERIMENT_TAG_NO_INDUCED_LANGUAGE_NO_COMPRESION_BASELINE_BOOTSTRAP] = build_experiment_no_induced_language_no_compression_baseline_bootstrap_primitives # DeepCoder style
    
    EXPERIMENTS_REGISTRY[EXPERIMENT_TAG_FULL_LANGUAGE_GENERATIVE_BOOTSTRAP] = build_experiment_full_language_generative_bootstrap_primitives # Full language with generative model.
    EXPERIMENTS_REGISTRY[EXPERIMENT_TAG_FULL_LANGUAGE_MUTUAL_EXCLUSIVITY_BOOTSTRAP] = build_experiment_full_language_mutual_exclusivity_bootstrap_primitives # Mutual exclusivity
    EXPERIMENTS_REGISTRY[EXPERIMENT_TAG_FULL_LANGUAGE_LANGUAGE_COMPRESSION_BOOTSTRAP] = build_experiment_full_language_language_compression_bootstrap_primitives # Language compression
    
    EXPERIMENTS_REGISTRY[EXPERIMENT_TAG_FULL_LANGUAGE_BOTH_ADD_ONS_BOOTSTRAP] = build_experiment_full_language_both_add_ons_bootstrap_primitives #ME + Language compression
    
    EXPERIMENTS_REGISTRY[EXPERIMENT_TAG_FULL_LANGUAGE_SYNTH_TRAIN_HUMAN_TEST] = build_experiment_full_language_synth_train_human_test_bootstrap
    EXPERIMENTS_REGISTRY[EXPERIMENT_TAG_FULL_LANGUAGE_HUMAN_TRAIN_HUMAN_TEST] = build_experiment_full_language_human_train_human_test_bootstrap

    EXPERIMENTS_REGISTRY[EXPERIMENT_TAG_BEST_DSL_NO_LANGUAGE_PURE_ENUMERATION] = build_experiment_best_dsl_no_language_pure_enumeration_bootstrap
    EXPERIMENTS_REGISTRY[EXPERIMENT_TAG_BEST_DSL_FULL_LANGUAGE_PURE_ENUMERATION_AND_NEURAL_SEARCH] = build_experiment_best_dsl_full_language_pure_enumeration_and_neural_search_bootstrap

    print(f"Registered a total of {len(EXPERIMENTS_REGISTRY)} experiments:")
    for experiment_name in EXPERIMENTS_REGISTRY:
        print(f"\t{experiment_name}")
    print("\n")    

def main(args):
    register_all_experiments()
    experiment_to_resume_checkpoint = optionally_generate_resume_command_for_log(args)
    cloud_launcher_command = build_cloud_launcher_command(args)
    experiment_commands = build_experiment_commands(args, experiment_to_resume_checkpoint)
    output_launch_commands_and_log_lines(cloud_launcher_command, experiment_commands, args)
    
if __name__ == '__main__':
  args = parser.parse_args()
  main(args) 
