"""
test_joint_models: Author : Catherine Wong

Tests joint model implementations. For now, a scratch interface for development.
"""
import sys

def test_joint_language_program_model(result, train_tasks, testing_tasks):
    # All tasks have a ground truth program and a name.
    
    for task in train_tasks:
        task_name = task.name
        task_language = result.taskLanguage[task_name]
        groundTruthProgram = task.groundTruthProgram 
        ground_truth_program_tokens = task.groundTruthProgram.left_order_tokens(show_vars=True) # A good example of how to turn programs into sequences as a baseline. Removes variables -- you could put this back. See program.py - line: 77
        print(f"Task name: {task_name}")
        print(f"Task language:  {task_language}")
        print(f"Ground truth program: {groundTruthProgram}")
        print(f"Ground truth program tokens: {ground_truth_program_tokens}")
        # Additional attributes on LOGO tasks: see makeLogoTasks.py
        # task.highresolution <== an array containing the image.
        sys.exit(0)