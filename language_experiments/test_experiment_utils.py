"""
test_experiment_utils.py | Author : Catherine Wong

Tests for the experiment_utils.py file. Tests must be added manually to the main.
"""
import experiment_utils 

TEST_DOMAIN_PREFIX = "test_domain"
TEST_EXPERIMENT_BASENAME = "test_experiment"
TEST_EXPERIMENT_LOG_DIRECTORY = "test_logs"
TEST_REGISTERED_CHECKPOINT = "test_checkpoint"
TEST_REGISTERED_CHECKPOINT_PATH = "test_registered_checkpoint_path_it=0_"
TEST_UNREGISTERED_CHECKPOINT_PATH = "TEST_UNREGISTERED_CHECKPOINT_PATH_it=0_"
TEST_CHECKPOINT_REGISTRY = {
    TEST_REGISTERED_CHECKPOINT : TEST_REGISTERED_CHECKPOINT_PATH
}

from types import SimpleNamespace as MockArgs

def test_get_experiment_argparser():
    parser = experiment_utils.get_experiment_argparser(domain_name_prefix=TEST_DOMAIN_PREFIX)
    args = parser.parse_args()
    
    assert (args.cloud_platform == experiment_utils.OM_FLAG)
    assert (args.number_random_replications == 1)
    assert (args.experiment_log_directory == experiment_utils.get_default_log_directory(TEST_DOMAIN_PREFIX)) 

def test_optionally_generate_resume_commands_checkpoint():
    mock_args = MockArgs(basename_to_resume=TEST_EXPERIMENT_BASENAME,
            generate_resume_command_for_log=None,
            generate_resume_command_for_checkpoint=TEST_REGISTERED_CHECKPOINT)
    maybe_experiments_to_resume = experiment_utils.optionally_generate_resume_commands(mock_args, TEST_CHECKPOINT_REGISTRY)
    
    assert TEST_EXPERIMENT_BASENAME in maybe_experiments_to_resume
    assert maybe_experiments_to_resume[TEST_EXPERIMENT_BASENAME] == TEST_REGISTERED_CHECKPOINT_PATH 

def optionally_generate_resume_command_for_log():
    """This is legacy code from the ICML submission. New usage should test it thoroughly."""
    pass

def test_optionally_generate_resume_command_for_checkpoint_in_registry():
    mock_args = MockArgs(basename_to_resume=TEST_EXPERIMENT_BASENAME,
            generate_resume_command_for_checkpoint=TEST_REGISTERED_CHECKPOINT)
    maybe_experiments_to_resume = experiment_utils.optionally_generate_resume_command_for_checkpoint(mock_args, TEST_CHECKPOINT_REGISTRY)
    
    assert TEST_EXPERIMENT_BASENAME in maybe_experiments_to_resume
    assert maybe_experiments_to_resume[TEST_EXPERIMENT_BASENAME] == TEST_REGISTERED_CHECKPOINT_PATH 

def test_optionally_generate_resume_command_for_checkpoint_unregistered():
    mock_args = MockArgs(basename_to_resume=TEST_EXPERIMENT_BASENAME,
            generate_resume_command_for_checkpoint=TEST_UNREGISTERED_CHECKPOINT_PATH)
    maybe_experiments_to_resume = experiment_utils.optionally_generate_resume_command_for_checkpoint(mock_args, TEST_CHECKPOINT_REGISTRY)
    
    assert TEST_EXPERIMENT_BASENAME in maybe_experiments_to_resume
    assert maybe_experiments_to_resume[TEST_EXPERIMENT_BASENAME] == TEST_UNREGISTERED_CHECKPOINT_PATH

def test_build_cloud_launcher_command_om():
    parser = experiment_utils.get_experiment_argparser(domain_name_prefix=TEST_DOMAIN_PREFIX)
    args = parser.parse_args()
    
    om_launcher_command = experiment_utils.build_cloud_launcher_command(args)
    print("\n\nVisual inspection test: is this the OM launcher command?")
    print(om_launcher_command)

# Mock registered experiment.
@experiment_utils.register_experiment(TEST_EXPERIMENT_BASENAME)
def build_experiment_test_experiment_basename(basename, args, experiment_to_resume_checkpoint):
    def experiment_parameters_fn():
        return  " --recognition_0 --recognition_1 language --Helmholtz 0 "
    return experiment_utils.build_experiment_command_information(basename, args, experiment_parameters_fn)

def test_build_experiment_commands_no_resume():
    mock_args = MockArgs(experiment_classes=[TEST_EXPERIMENT_BASENAME],
                        cloud_platform=experiment_utils.OM_FLAG,
                        number_random_replications=1,
                        experiment_prefix=TEST_DOMAIN_PREFIX)
    all_experiment_commands = experiment_utils.build_experiment_commands(mock_args, experiment_to_resume_checkpoint={})
    print("\n\nVisual inspection test: is this the dictionary command?")
    print(all_experiment_commands)

def test_build_experiment_commands_with_resume():
    mock_args = MockArgs(experiment_classes=[TEST_EXPERIMENT_BASENAME],
                        cloud_platform=experiment_utils.OM_FLAG,
                        number_random_replications=1,
                        experiment_prefix=TEST_DOMAIN_PREFIX,
                        basename_to_resume=TEST_EXPERIMENT_BASENAME,
                        generate_resume_command_for_log=None,
                                generate_resume_command_for_checkpoint=TEST_REGISTERED_CHECKPOINT)
    
    maybe_experiments_to_resume = experiment_utils.optionally_generate_resume_commands(mock_args, TEST_CHECKPOINT_REGISTRY)
    all_experiment_commands = experiment_utils.build_experiment_commands(mock_args, experiment_to_resume_checkpoint=maybe_experiments_to_resume)
    print("\n\nVisual inspection test: is this the dictionary command?")
    print(all_experiment_commands)

def test_generate_all_launch_commands_and_log_lines():
    mock_args = MockArgs(experiment_classes=[TEST_EXPERIMENT_BASENAME],
                        cloud_platform=experiment_utils.OM_FLAG,
                        number_random_replications=1,
                        experiment_prefix=TEST_DOMAIN_PREFIX,
                        basename_to_resume=TEST_EXPERIMENT_BASENAME,
                        experiment_log_directory=TEST_EXPERIMENT_LOG_DIRECTORY,
                        generate_resume_command_for_log=None,
                        generate_resume_command_for_checkpoint=TEST_REGISTERED_CHECKPOINT)
    experiment_utils.generate_all_launch_commands_and_log_lines(mock_args, checkpoint_registry=TEST_CHECKPOINT_REGISTRY)
    
    

def main():
    # test_get_experiment_argparser()
    # test_optionally_generate_resume_commands_checkpoint()
    # test_optionally_generate_resume_command_for_checkpoint_in_registry()
    # test_optionally_generate_resume_command_for_checkpoint_unregistered()
    # test_build_cloud_launcher_command_om()
    
    # test_build_experiment_commands_no_resume()
    # test_build_experiment_commands_with_resume()
    
    test_generate_all_launch_commands_and_log_lines()
if __name__ == '__main__':
    main()