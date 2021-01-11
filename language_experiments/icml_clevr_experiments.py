"""
icml_clevr_experiments.py | Author: Catherine Wong.

This generates the command-line output necessary to launch experiments on cloud platforms, such as OpenMind, Azure, and Google Cloud.

Example usage:
python icml_clevr_experiments.py 
    --cloud_platform om
    --number_random_replications
    --experiment_prefix clevr
    --experiment_classes all
    --output_all_commands_at_once

Available experiments: TODO
All experiments must be manually added to an experiment registry in register_all_experiment.
"""

DEFAULT_CLEVR_DOMAIN_NAME_PREFIX = 'clevr'
DEFAULT_LOG_DIRECTORY = f"../ec_language_logs/{DEFAULT_CLEVR_DOMAIN_NAME_PREFIX}"
DEFAULT_PYTHON_MAIN_COMMAND = f"python bin/clevr.py "

# Default parameters for running on OpenMind
OM_FLAG = 'om'
OM_SCP_COMMAND = "zyzzyva@openmind7.mit.edu" # Specific to Catherine Wong
DEFAULT_OM_CPUS_PER_TASK = 12
DEFAULT_OM_MEM_PER_TASK = 30000
DEFAULT_OM_TIME_PER_TASK = 10000

# Default parameters for each experiment.
DEFAULT_TASK_BATCH_SIZE = 25
DEFAULT_RECOGNITION_STEPS = 10000
DEFAULT_TEST_EVERY = 3
DEFAULT_ITERATIONS = 10
DEFAULT_ENUMERATION_TIMEOUT = 3600

# Tags for experiment classes.
EXPERIMENTS_REGISTRY = dict()
EXPERIMENT_TAG_NO_LANGUAGE_BASELINE_BOOTSTRAP = 'no_language_baseline_bootstrap'
EXPERIMENT_TAG_FULL_LANGUAGE_GENERATIVE_BOOTSTRAP = 'full_language_generative_bootstrap'

GENERATE_ALL_FLAG = 'all'
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
    if args.cloud_platform == OM_FLAG:
        logfile_basename = os.path.basename(remote_logfile_path)
        local_logfile_path = os.path.join(args.experiment_log_directory, logfile_basename)
        scp_command = f"scp {OM_SCP_COMMAND}:{remote_logfile_path} {local_logfile_path}"
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
        checkpoints_to_resume = [checkpoint for checkpoint in checkpoints if checkpoint_iteration_tag] 
        if len(checkpoints_to_resume) != 1:
            print("Did not find a matching checkpoint. Exiting")
            sys.exit(0)
        checkpoint_to_resume = checkpoints_to_resume[0]
    print(f"Found checkpoint {checkpoint_to_resume} to resume.")
    return checkpoint_to_resume

def generate_timestamped_record_for_csv_logs(args, experiment_name, all_final_commands, experiment_information_dict):
    """Generates a record of these experiments for the CSV logs."""
    experiment_tags = input(f"Comma-separated experiment-level tags? (Default: None)") or ""
    # Builds a list based on the current spreadsheet layout.
    csv_log_entries_for_experiment = []
    for replication_idx, replication_full_command in enumerate(all_final_commands):
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
        for replication_idx, replication_command in enumerate(replication_commands):
            final_command = build_final_command_from_launcher_and_experiment(cloud_launcher_command, experiment_name, replication_command, replication_idx, resume_command)
            all_final_commands.append(final_command)
            print(final_command)
        print("\n")
        generate_timestamped_record_for_csv_logs(args, experiment_name, all_final_commands, experiment_information_dict)
        if not args.output_all_commands_at_once:
            input("....hit enter for next experiments\n")
            
def build_final_command_from_launcher_and_experiment(cloud_launcher_command, experiment_name, replication_command, replication_idx, resume_command):
    """Builds the final launch command from a given experimental command and its cloud launcher."""
    if args.cloud_platform == OM_FLAG:
        formatted_launch_command = cloud_launcher_command.format(experiment_name, replication_idx, experiment_name, replication_idx)
        return formatted_launch_command + replication_command + resume_command + " &"
    else:
        print(f"Unknown cloud platform: {args.cloud_platform}")
        sys.exit(0)
    
def build_cloud_launcher_command(args):
    """Builds the cloud launcher command for a given cloud platform, with additional prompts if needed.
    """
    if args.cloud_platform == OM_FLAG:
        return build_om_launcher_command(args)
    else:
        print(f"Unknown cloud platform: {args.cloud_platform}")
        sys.exit(0)
        
def build_om_launcher_command(args):
    """Builds the launcher command for running on OpenMind. 
    Returns a string command that can be run """
    print("Running on OpenMind. Please input the following parameters:")
    number_cpus_per_task = input(f"Number of CPUS per task? (Default: {DEFAULT_OM_CPUS_PER_TASK})") or DEFAULT_OM_CPUS_PER_TASK
    memory_per_cpu = input(f"Memory per task? (Default: {DEFAULT_OM_MEM_PER_TASK})") or DEFAULT_OM_MEM_PER_TASK
    om_base_command = f"srun --job-name="+args.experiment_prefix+"-language-{}_{} --output="+args.experiment_log_directory+"/" +args.experiment_prefix+"-{}_{} --ntasks=1 --mem-per-cpu="+str(memory_per_cpu)+" --gres=gpu --cpus-per-task "+str(number_cpus_per_task)+" --time="+ str(DEFAULT_OM_TIME_PER_TASK) + ":00 --qos=tenenbaum --partition=tenenbaum singularity exec -B /om2  --nv ../dev-container.img "
    print("\n")
    return om_base_command

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
        experiment_name, experiment_information_dict = experiment_command_builder_fn(experiment_class, args)
        add_resume_commands(experiment_class, experiment_information_dict, experiment_to_resume_checkpoint)
        experiment_commands_dict[experiment_name] = experiment_information_dict
    return experiment_commands_dict

def add_resume_commands(experiment_class, experiment_information_dict, experiment_to_resume_checkpoint):
    """
    Adds the resume command to the experiment information_dict if we have found an appropriate checkpoint.
    Mutates: experiment_information_dict
    """
    experiment_information_dict['resume_command'] = " "
    if experiment_class in experiment_to_resume_checkpoint:
        checkpoint_to_resume = experiment_to_resume_checkpoint[experiment_class]
        experiment_information_dict['resume_command'] = f" --resume {checkpoint_to_resume} "
        
def build_replication_commands(experiment_command, args):
    """Takes a basic experiment class command and builds a set of replication commands for it. Returns : [array of experiment_replication_commands]"""
    return [
        experiment_command + f' --taskReranker randomShuffle --seed {replication_idx} '
        for replication_idx in range(1, args.number_random_replications + 1)
    ]
    
def get_interactive_experiment_parameters():
    """Prompts the user for interactive experiment parameters, which vary for most experiments. 
    Returns a set of experiment parameters as a command.
    A string tag that distinguishes this particular experiment"""
    task_datasets = input(f"Tasks to test on? (Default: {GENERATE_ALL_FLAG})") or GENERATE_ALL_FLAG
    task_batch_size = input(f"Task batch size? (Default: {DEFAULT_TASK_BATCH_SIZE})") or DEFAULT_TASK_BATCH_SIZE
    number_of_iterations = input(f"Number of iterations? (Default: {DEFAULT_ITERATIONS})") or DEFAULT_ITERATIONS
    test_every = input(f"Test on every N iterations? (Default: {DEFAULT_TEST_EVERY})") or DEFAULT_TEST_EVERY
    enumeration_timeout = input(f"Enumeration timeout per task? Same as testing timeout. (Default: {DEFAULT_ENUMERATION_TIMEOUT})") or DEFAULT_ENUMERATION_TIMEOUT
    
    interactive_experiment_parameters = f"--enumerationTimeout {enumeration_timeout} --testingTimeout {enumeration_timeout} --iterations {number_of_iterations} --taskBatchSize {task_batch_size} --testEvery {test_every} --taskDatasets {task_datasets} "
    
    interactive_experiment_tag = f"et_{enumeration_timeout}_it_{number_of_iterations}_batch_{task_batch_size}"
    return interactive_experiment_parameters, interactive_experiment_tag  

def get_shared_experiment_parameters():
    """Gets parameters that are shared across all experiments. Returns a set of experiment parameters as a command."""
    return f"--biasOptimal --contextual --no-cuda --skip_first_test --moses_dir ./moses_compiled --smt_phrase_length 1 --language_encoder recurrent --recognitionSteps {DEFAULT_RECOGNITION_STEPS} "

def build_experiment_baseline_bootstrap_primitives(basename, args):
    """Builds the baseline experiments: these run DreamCoder without any language in the loop. Uses bootstrap primitives.
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
    experiment_class_parameters = "--recognition_0 examples --Helmholtz 0.5 --primitives clevr_bootstrap clevr_map_transform"
    
    experiment_command = DEFAULT_PYTHON_MAIN_COMMAND + interactive_experiment_parameters + shared_experiment_parameters + experiment_class_parameters
    
    all_replication_commands = build_replication_commands(experiment_command, args)
    experiment_name = f"{basename}-{interactive_experiment_tag}"
    experiment_information_dict = {
        'replication_commands' : all_replication_commands,
        'experiment_basename' : basename,
        'interactive_parameters' : interactive_experiment_parameters,
    }
    return experiment_name, experiment_information_dict

def build_experiment_full_language_generative_bootstrap_primitives(basename, args):
    """Builds the basic full language experiment: this uses language, but not any pseudoalignments or language-guided compression. Uses bootstrap primitives.
    Returns: 
        full_experiment_name, [array of experiment replication commands]
    """
    print(f"Building an experiment for class: {basename}.")
    
    interactive_experiment_parameters, interactive_experiment_tag  = get_interactive_experiment_parameters() 
    shared_experiment_parameters = get_shared_experiment_parameters()
    experiment_class_parameters = "--recognition_0 --recognition_1 examples language --Helmholtz 0.5 --primitives clevr_bootstrap clevr_map_transform  --synchronous_grammar --lc_score 0 "
    
    experiment_command = DEFAULT_PYTHON_MAIN_COMMAND + interactive_experiment_parameters + shared_experiment_parameters + experiment_class_parameters
    
    all_replication_commands = build_replication_commands(experiment_command, args)
    experiment_name = f"{basename}-{interactive_experiment_tag}"
    experiment_information_dict = {
        'replication_commands' : all_replication_commands,
        'experiment_basename' : basename,
        'interactive_parameters' : interactive_experiment_parameters,
    }
    return experiment_name, experiment_information_dict
    

def register_all_experiments():
    """Adds functions for a given experiment type to a global registry.
    Mutates: EXPERIMENTS_REGISTRY
    """
    print("Registering experiments for CLEVR...")
    EXPERIMENTS_REGISTRY[EXPERIMENT_TAG_NO_LANGUAGE_BASELINE_BOOTSTRAP] = build_experiment_baseline_bootstrap_primitives # Baseline experiment.
    EXPERIMENTS_REGISTRY[EXPERIMENT_TAG_FULL_LANGUAGE_GENERATIVE_BOOTSTRAP] = build_experiment_full_language_generative_bootstrap_primitives # Full language with generative model.
    
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
