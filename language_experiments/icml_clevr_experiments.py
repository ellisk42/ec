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
OM_FLAG = 'om'

DEFAULT_OM_CPUS_PER_TASK = 12
DEFAULT_OM_MEM_PER_TASK = 15000

EXPERIMENTS_REGISTRY = dict()
EXPERIMENT_TAG_NO_LANGUAGE_BASELINE = 'no_language_baseline'

GENERATE_ALL_FLAG = 'all'
import sys
import argparse
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
                    required=True,
                    nargs='*',
                    help="Which experiments to run. 'all' for all of the ones currently in the registry.")
parser.add_argument('--experiment_log_directory',
                    default=DEFAULT_LOG_DIRECTORY,
                    help="The logging output directory to which we will output the logging information.")
parser.add_argument('--output_all_commands_at_once',
                    action='store_true',
                    help="If true, we print all commands at once, rather than iteratively by experiment type.")

def generate_timestamped_record_for_csv_logs():
    """Generates a record of experiments for the CSV logs."""
    pass

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
    om_base_command = f"srun --job-name="+args.experiment_prefix+"-language-{} --output="+args.experiment_log_directory+"/{} --ntasks=1 --mem-per-cpu="+memory_per_cpu+" --gres=gpu --cpus-per-task "+number_cpus_per_task+" --time=10000:00 --qos=tenenbaum --partition=tenenbaum singularity exec -B /om2  --nv ../dev-container.img "
    print("\n")
    return om_base_command

def build_experiment_commands(args):
    """Given a set of experiment tags to run, builds the appropriate commands from the registry, including their replications.
    
    Returns: dict {
        full_experiment_name : [array of experiment_replication_commands]
    }
    """
    experiment_classes = args.experiment_classes
    if args.experiment_classes == [GENERATE_ALL_FLAG]:
        experiment_classes = EXPERIMENTS_REGISTRY.keys()
    experiment_commands_dict = defaultdict(list)
    for experiment_class in experiment_classes:
        if experiment_class not in EXPERIMENTS_REGISTRY:
            print(f"Not found in the experiments registry: {experiment_class}")
            sys.exit(0)
        experiment_command_builder_fn = EXPERIMENTS_REGISTRY[experiment_class]
        experiment_name, experiment_commmands = experiment_command_builder_fn()
        experiment_classes[experiment_name] = experiment_commands
    return experiment_commands_dict

def output_launch_commands_and_log_lines(cloud_launcher_command, experiment_commands, args):
    """Outputs the launch commands for a set of experiments, and a comma separated line that can be logged in an experiment spreadsheet."""
    pass
        
    
def build_experiment_baseline_bootstrap_primitives(args):
    """Builds the baseline experiments: these run DreamCoder without any language in the loop. Uses bootstrap primitives.
    Returns: 
        full_experiment_name, [array of experiment replication commands]
    """
    pass

def register_all_experiments():
    """Adds functions for a given experiment type to a global registry.
    Mutates: EXPERIMENTS_REGISTRY
    """
    print("Registering experiments for CLEVR...")
    EXPERIMENTS_REGISTRY[EXPERIMENT_TAG_NO_LANGUAGE_BASELINE] = build_experiment_baseline_bootstrap_primitive
    
    print(f"Registered a total of {len(EXPERIMENTS_REGISTRY)} experiments:")
    for experiment_name in EXPERIMENTS_REGISTRY:
        print(f"\t{experiment_name}")
    print("\n")

def main(args):
    register_all_experiments()
    cloud_launcher_command = build_cloud_launcher_command(args)
    experiment_commands = build_experiment_commands(args)
    output_launch_commands_and_log_lines(cloud_launcher_command, experiment_commands, args)
    
if __name__ == '__main__':
  args = parser.parse_args()
  main(args) 