"""
experiment_utils.py | Author : Catherine Wong.

Common utilities for launching cloud-based experiments for DreamCoder.
Largely contains utilities for generating command line arguments that can be run with various cloud computing structures, as well as common constants for these kinds of experiments.

Commonly used utilities are:
    parser = get_experiment_argparser(domain_name_prefix) -- gets an argparser with the commmands.
    @register_experiment(name) -- registers experiment builder functions for the experiment commands.
    
    generate_all_launch_commands_and_log_lines(args, checkpoint_registry=dict()) -- takes a default checkpoint registry and generates all launch commands and log_lines.
"""
import os
import sys
import subprocess
import argparse
import datetime
from collections import defaultdict

# EC specific constants.
DEFAULT_LANGUAGE_LOGS_DIR = "../ec_language_logs"
DEFAULT_CHECKPOINTS_DIR = "../experimentOutputs"

GENERATE_ALL_FLAG = "all"

# Default parameters for each experiment.
DEFAULT_TASK_BATCH_SIZE = 400
DEFAULT_RECOGNITION_STEPS = 10000
DEFAULT_TEST_EVERY = 1
DEFAULT_ITERATIONS = 10
DEFAULT_ENUMERATION_TIMEOUT = 720
DEFAULT_MEM_PER_ENUMERATION_THREAD = 5000000000 # 5 GB
DEFAULT_PSEUDOALIGNMENTS_WEIGHT = 0.05
DEFAULT_LANGUAGE_COMPRESSION_WEIGHT = 0.05
DEFAULT_LANGUAGE_COMPRESSION_MAX_COMPRESSION = 5 

## Cloud computing constants.
# Default parameters for running on OpenMind
OM_FLAG = 'om'
OM_SCP_COMMAND = "zyzzyva@openmind7.mit.edu" # Specific to Catherine Wong
NUM_CPUS_TAG = 'CPUs'

DEFAULT_SLURM_CPUS_PER_TASK = 24
DEFAULT_SLURM_TIME_PER_TASK = 10000
DEFAULT_SLURM_MEMORY = "50G"  # "MaxMemPerNode"

# Default parameters for running on Supercloud
SUPERCLOUD_FLAG = 'supercloud'
SUPERCLOUD_SCP_COMMAND = "zyzzyva@txe1-login.mit.edu" # Specific to Catherine Wong

# Default parameters for running on Azure.
AZURE_FLAG = 'azure'
AZ_SCP_COMMAND = "zyzzyva@MUST FILL IN"
AZURE_RESOURCE_GROUP = "resource-group"
AZURE_LOCATION = "location"
AZURE_IMAGE = "image"
AZURE_DEFAULT_MACHINE_TYPE = "Standard_E48s_v3"
AZURE_DEFAULT_RESOURCES_0 = { 
    AZURE_RESOURCE_GROUP : "zyzzyva-clevr-0", # Example. This should be changed.
    AZURE_LOCATION : "eastus",
    AZURE_IMAGE : "myGallery/zyzzyva-clevr-0-base-image/0.0.1"
}
AZURE_RESOURCES = {
    AZURE_DEFAULT_RESOURCES_0[AZURE_RESOURCE_GROUP] : AZURE_DEFAULT_RESOURCES_0,
}

def generate_all_launch_commands_and_log_lines(args, checkpoint_registry=dict()):
    """Importable public utility function for generating launch commands and log lines.
    Runs a full pipeline of launch commands.
    """
    experiment_to_resume_checkpoint = optionally_generate_resume_commands(args, checkpoint_registry=checkpoint_registry)
    cloud_launcher_command = build_cloud_launcher_command(args)
    experiment_commands = build_experiment_commands(args, experiment_to_resume_checkpoint)
    output_launch_commands_and_log_lines(cloud_launcher_command, experiment_commands, args)

EXPERIMENTS_REGISTRY = {}
def register_experiment(name):
    """Importable decorator. Defines the @register_experiment(name) decorator for a global checkpoints registry."""
    return make_registry_decorator(name, registry=EXPERIMENTS_REGISTRY)
    
def get_experiment_argparser(domain_name_prefix):
    """Importable public utitlity function. Returns a default experiment parser for running experiments.
    domain_name_prefix: string prefix for this domain.
    
    Supported default usage:
        --experiment_classes list of experiment tags for this domain
        --cloud_platform om | azure | supercloud
        --number_random_replications 1
        --experiment_prefix DOMAIN_NAME
        --generate_resume_command_for_log <filepath or registry key>
        --generate_resume_command_for_checkpoint <filepath or registry key>
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_classes',
                    nargs='*',
                    default=[],
                    help="Which experiments to run. 'all' for all of the ones currently in the registry.")
    parser.add_argument('--cloud_platform',
                    default=OM_FLAG,
                    choices=[OM_FLAG,AZURE_FLAG,SUPERCLOUD_FLAG],
                    help='Which cloud platform you are attempting to generate commands for.')
    parser.add_argument('--number_random_replications',
                    default=1,
                    type=int,
                    help='The total number of replications to run for a given experiment type.')
    
    parser.add_argument('--experiment_prefix',
                    default=domain_name_prefix,
                    help='The experimental prefix that will be appended to the experiment.')
    parser.add_argument('--experiment_log_directory',
                    default=get_default_log_directory(domain_name_prefix),
                    help="The logging output directory to which we will output the logging information.")
    
    parser.add_argument('--basename_to_resume',
                    help="If provided, overrides the automatically derived basename for the resumed checkpoints.")
    parser.add_argument('--generate_resume_command_for_log',
                    help="String path or a registry key for a logfile to generate a resume command for. Looks for a log in the cached logs, otherwise SCPS it.")
    parser.add_argument('--generate_resume_command_for_checkpoint',
                    help="String path or a registry key for a checkpoint to generate a resume command for. Looks for a checkpoint in the cached checkpoints, otherwise SCPS it.") 
    parser.add_argument('--no_laps',
                    help="Whether to generate commands for the original EC.",
                    action="store_true")    
    return parser
    
    

def make_registry_decorator(name, registry):
    """Utility to define registry decorators for a given registry dict."""
    def wrapper(f):
        registry[name] = f
        return f
    return wrapper

def get_default_log_directory(domain_name_prefix):
    return f"{DEFAULT_LANGUAGE_LOGS_DIR}/{domain_name_prefix}"

## Find checkpoints for resuming experiments.
def optionally_generate_resume_commands(args, checkpoint_registry={}):
    """Gets checkpoints for resuming experiments. Expects either DreamCoder logfiles, or a checkpoint path.
    Expects checkpoint_registry = dictionary of checkpoint files.
    
    Returns: {experiment_basename_to_resume : checkpoint}
    """
    if not args.generate_resume_command_for_log and not args.generate_resume_command_for_checkpoint: return dict()
    if args.generate_resume_command_for_checkpoint:
        return optionally_generate_resume_command_for_checkpoint(args, checkpoint_registry)
    if args.generate_resume_command_for_log:
        return optionally_generate_resume_command_for_log(args, checkpoint_registry)

def optionally_generate_resume_command_for_checkpoint(args, checkpoint_registry):
    """
    Generates a resume command for a checkpoint either with a filepath or by looking it up in the registry.
    Requires args.experiment_basename
    Returns: {experiment_basename_to_resume : checkpoint}
    """
    if not args.basename_to_resume:
        print("Resume from checkpoint:requires --basename_to_resume .")
        sys.exit(0)
    
    experiment_basename = args.basename_to_resume 
    checkpoint_to_resume = args.generate_resume_command_for_checkpoint
    print(f"Going to attempt to generate a resume command for checkpoint. This should be an absolute path: {checkpoint_to_resume}")
    if checkpoint_to_resume in checkpoint_registry:
        checkpoint_to_resume = checkpoint_registry[checkpoint_to_resume]
        print(f"Found checkpoint in registry: {checkpoint_to_resume}")

    return {experiment_basename : checkpoint_to_resume}

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
    experiment_basename = args.basename_to_resume if args.basename_to_resume else get_experiment_basename_from_logfile_path(local_logfile_path)
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

## Utility functions for managing global experiment parameters.
GLOBAL_EXPERIMENTS_ARGUMENTS = dict() # Tracks globally set arguments.
USER_INPUT_DEFAULT_PARAMETERS = dict() # Tracks the latest user input parameter so we can offer it as a default.

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

## Build wrapper commands for cloud computing.
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
    mem_per_task = get_input_or_default("Max mem per node?", DEFAULT_SLURM_MEMORY)
    # This requires limitation on the program side, so we set it in the global dictionary.
    GLOBAL_EXPERIMENTS_ARGUMENTS[NUM_CPUS_TAG] = number_cpus_per_task
    slurm_base_command = f"sbatch --job-name="+args.experiment_prefix+"-language-{}_{} --output="+args.experiment_log_directory+"/" +args.experiment_prefix+"-{}_{} --ntasks=1" + f" --mem={mem_per_task} --cpus-per-task "+str(number_cpus_per_task)
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


## Utility functions to build new experiment commands for DreamCoder-style experiments.
def get_default_python_command(args):
    return f"python bin/{args.experiment_prefix}.py "

def build_experiment_commands(args, experiment_to_resume_checkpoint):
    """Given a set of experiment tags to run, builds the appropriate commands from the registry, including their replications.
    
    experiment_to_resume_checkpoint is of the form {experiment_name_to_resume : checkpoints to resume}
    
    Expects experiment_command_builder_fns in the EXPERIMENT_REGISTRY. See build_experiment_command_information.
    
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
    
def get_interactive_experiment_parameters():
    """Prompts the user for interactive experiment parameters, which vary for most experiments. 
    Returns a set of experiment parameters as a command.
    A string tag that distinguishes this particular experiment"""
    task_batch_size = get_input_or_default("Task batch size?", DEFAULT_TASK_BATCH_SIZE)
    number_of_iterations = get_input_or_default("Number of iterations?", DEFAULT_ITERATIONS)
    test_every = get_input_or_default("Test on every N iterations?", DEFAULT_TEST_EVERY)
    enumeration_timeout = get_input_or_default("Enumeration timeout per task?", DEFAULT_ENUMERATION_TIMEOUT)
    testing_timeout = get_input_or_default("Testing timeout per task?", enumeration_timeout)
    recognition_steps = get_input_or_default("Total recognition steps? ", DEFAULT_RECOGNITION_STEPS)
    rep_tag = get_input_or_default("Add any replication tag to avoid namespace collision? ", "")
    
    interactive_experiment_parameters = f"--enumerationTimeout {enumeration_timeout} --testingTimeout {testing_timeout} --iterations {number_of_iterations} --taskBatchSize {task_batch_size} --testEvery {test_every} --recognitionSteps {recognition_steps} "
    
    interactive_experiment_tag = f"et_{enumeration_timeout}_it_{number_of_iterations}_batch_{task_batch_size}"
    if len(rep_tag) > 0:
        interactive_experiment_tag += f"_{rep_tag}"
    return interactive_experiment_parameters, interactive_experiment_tag  

def get_shared_experiment_parameters(args):
    """Gets parameters that are shared across all language experiments. Returns a set of experiment parameters as a command."""
    max_mem_per_enumeration_thread = get_input_or_default("Maximum memory per enumeration thread?", DEFAULT_MEM_PER_ENUMERATION_THREAD)
    
    # Add any global parameters.
    global_parameters_command = " ".join([f"--{global_param} {global_param_value} " for (global_param, global_param_value) in GLOBAL_EXPERIMENTS_ARGUMENTS.items()])
    
    if args.no_laps:
        preload_frontiers = get_input_or_default("Preload frontiers file?", "NONE")
        
        return f"--biasOptimal --contextual --no-cuda --preload_frontiers {preload_frontiers}  " + global_parameters_command
    else:
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
    shared_experiment_parameters = get_shared_experiment_parameters(args)
    experiment_class_parameters = experiment_parameters_fn()
    experiment_command = get_default_python_command(args) + interactive_experiment_parameters + shared_experiment_parameters + experiment_class_parameters
    
    all_replication_commands = build_replication_commands(experiment_command, args)
    experiment_name = f"{basename}-{interactive_experiment_tag}"
    experiment_information_dict = {
        'replication_commands' : all_replication_commands,
        'experiment_basename' : basename,
        'interactive_parameters' : interactive_experiment_parameters,
    }
    return experiment_name, experiment_information_dict


## Utility functions to output the full launch commands.
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
            final_command = build_final_command_from_launcher_and_experiment(args, cloud_launcher_command, experiment_name, replication_command, replication_idx, resume_command, is_random=is_random)
            all_final_commands.append(final_command)
            print(final_command)
            print("\n")
        generate_timestamped_record_for_csv_logs(args, experiment_name, all_final_commands, experiment_information_dict)
        
        input("....hit enter for next experiments\n")
            
def build_final_command_from_launcher_and_experiment(args, cloud_launcher_command, experiment_name, replication_command, replication_idx, resume_command, is_random=False):
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