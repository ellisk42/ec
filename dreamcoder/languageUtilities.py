"""
languageUtilities.py | Author: Catherine Wong 
"""
import json
import numpy as np
from dreamcoder.frontier import *
from copy import deepcopy
from collections import defaultdict
import nltk.translate.bleu_score as bleu_score

import seaborn as sns

def generate_independent_tasks_for_language_descriptions(result, tasks, testingTasks):
    """
    Updates the ECResult object so that the tasks themselves are different depending on each description. There will be only one description per task.
    
    Mutates: result.taskLanguage; result.taskSolutions; result.allFrontiers; 
    Returns tasks, testingTasks
    """
    print(f"Generating independent tasks for each language description. Initial tasks are {len(tasks)} train | {len(testingTasks)} test.")
    
    all_new_task_names = set()
    new_tasks = {
        "train" : [],
        "test" : []
    }
    new_train_tasks, new_test_tasks = [], []
    for (split, tasks) in [("train", tasks), ("test", testingTasks)]:
        for original_task in tasks:
            if original_task.name not in result.taskLanguage:
                all_new_task_names.add(original_task.name)
                new_tasks[split].append(original_task)
            else:
                for description in result.taskLanguage[original_task.name]:
                    singular_description = [description] # Singleton array for description.
                    new_task_name = f"{original_task.name}_{description}"
                    if new_task_name not in all_new_task_names:
                        new_task = deepcopy(original_task)
                        new_task.name = new_task_name
                    # Add the new task description.
                    result.taskLanguage[new_task_name] = singular_description
                    all_new_task_names.add(new_task_name)
                    new_tasks[split].append(new_task)
    # Update the task language.
    result.taskLanguage = {
        task_name : result.taskLanguage[task_name] for task_name in result.taskLanguage if task_name in all_new_task_names
    }
    
    result.taskSolutions={t: Frontier([],task=t) for t in new_tasks["train"]}
    result.numTestingTasks=len(new_tasks["test"])
    result.allFrontiers={t: Frontier([],task=t) for t in  new_tasks["train"]}

    print(f"Generated new independent tasks: {len(new_tasks['train'])} train | {len(new_tasks['test'])} test.")
    return new_tasks["train"],  new_tasks["test"]

def languageForTasks(languageDataset, languageDatasetDir, taskDict):
    """
    Loads a language dataset from {languageDatasetDir}/{languageDataset}.
    taskDict: {task_name : []}
    
    If not using the backward compatible languageForPathNameTasks, this assumes there are {train, test} splits, and loads a separate vocabulary for each one.
    taskDict fully determines all the tasks we will load, and does not obey train/test splits.
    Returns {task_name : list of sentences for each task, for the tasks in the taskDict.}
    
    """
    if type(languageDataset) is not list:
        languageDataset = [languageDataset]
        
    import os
    import json
    if 'path' in languageDataset[0]:
        return languageForPathNameTasks(languageDataset, languageDatasetDir, taskDict)
    
    vocabularies = {"train": set(), "test": set()}
    from pathlib import Path
    for dataset in languageDataset:
        dataset_path = os.path.join(languageDatasetDir, dataset)
        for split in ("train", "test"):
            try:
                split_path = os.path.join(dataset_path, split)
                with open(os.path.join(split_path, "language.json"), 'rb') as f:
                    languageData = json.load(f)
                    for t_name in taskDict:
                        if t_name in languageData:
                            taskDict[t_name] = languageData[t_name]
                with open(os.path.join(split_path, "vocab.json"), 'rb') as f:
                    vocabularies[split].update(json.load(f))
            except Exception as e:
                print(e)
                print(f"Not found: dataset for {split_path}")
                continue
    
    for k in vocabularies:
        vocabularies[k] = sorted(list(vocabularies[k]))

    print("Found language for {}/{} tasks".format(len([t for t in taskDict if len(taskDict[t]) > 0]), len(taskDict)))
    print(f"Found vocabularies of n=[{len(vocabularies['train'])}] for train and n=[{len(vocabularies['test'])}] for test.")
    
    return taskDict, vocabularies
    
def languageForPathNameTasks(languageDataset, taskDict):
    """
    Backwards compatability with language datasets that were keyed by a stimuli path name.
    """
    import json
    import os.path as osp
    
    with open(languageDataset, 'r') as f:
        languageDataset = json.load(f)
    
    def extract_task_name(task_path):
        return osp.basename(task_path).split('name_')[-1].split('.')[0]
    languageDataset = {
        extract_task_name(task_path) : [data[0] for data in languageData]
        for (task_path, languageData) in languageDataset.items()
    }
    
    def task_to_safe_name(task_name): 
        return task_name.replace("=","_").replace(' ','_').replace('/','_').replace("-","_").replace(")","_").replace("(","_")
    
    # Check for collisions in safe name.
    from collections import defaultdict
    task_safe_names = defaultdict(list)
    for t in taskDict:
        task_safe_names[task_to_safe_name(task_name )].append(task_name)
        
    languageForTasks = {}
    for t in taskDict:
        task_safe_name = task_to_safe_name(task_name)
        if task_safe_name not in languageDataset or len(languageDataset[task_safe_name]) < 1:
            print("Missing language for : {}".format(task_name))
        if len(task_safe_names[task_safe_name]) > 1:
            print("Collision: safe-name: {}, names: {}".format(task_safe_name, task_safe_name[task_safe_name]))
        language_for_task = languageDataset[task_safe_name] if task_safe_name in languageDataset else []
        languageForTasks[t] = language_for_task
    print("Found language for {}/{} tasks".format(len([t for t in languageForTasks if len(languageForTasks[t]) > 0]), len(languageForTasks)))
    return languageForTasks, None
        

NOISY_DATA_ANALYSIS_FILE = "noise_error_analysis.json"

SPLIT_TAG = "SPLIT"
TRAIN, TEST = "train", "test"
LANGUAGE_DESCRIPTIONS_TAG = "language_descriptions"
SEMANTIC_INCORRECTNESS_TAG = "semantic_incorrectness"
AVERAGE_SEMANTIC_INCORRECTNESS = "average_semantic_correctness"
SEMANTIC_CORRECT, SEMANTIC_PARTIAL_CORRECT, SEMANTIC_INCORRECT = "semantic_correct", "semantic_partial_correct", "semantic_incorrect"

BLEU_SCORE_TAG = "bleu_score"
INTERANNOTATOR_BLEU = "interannotator_bleu"
SYNTHETIC_TO_HUMAN_BLEU = "synthetic_to_human_bleu"
AVERAGE_INTERANNOTATOR_BLEU = "average_interannotator_bleu"
AVERAGE_SYNTHETIC_TO_HUMAN_BLEU = "average_synthetic_bleu"

GROUND_TRUTH_LOG_LIKELIHOOD_TAG = "ground_truth_log_likelihood"
def run_language_data_analysis_label_noisy_data(result, languageDataset, languageDatasetDir):
    """Runs an interactive thread to label noisy data.
    Saves to {languageDatasetDir}/{languageDataset[0]}/{NOISY_DATA_ANALYSIS_FILE}.
    """
    # Load existing data or create.
    noisy_data_analysis_filepath, noisy_data_analysis_object = load_or_create_noisy_data_analysis_file(result, languageDataset, languageDatasetDir) 
    
    noisy_data_analysis_object = interactive_labeling_for_noise_analysis(result, noisy_data_analysis_object)
    
    print(f"Writing out final labeled data to {noisy_data_analysis_filepath}")
    with open(noisy_data_analysis_filepath, "w") as f:
        json.dump(noisy_data_analysis_object, f)
    sys.exit(0)

def load_or_create_noisy_data_analysis_file(result, languageDataset, languageDatasetDir):
    """
    Loads noisy language data or creates a file containing the 
    Returns: 
        filepath to file.
        noisy_data_analysis_object: {language_object}
    """
    noisy_data_analysis_filepath = os.path.join(languageDatasetDir, languageDataset[0], NOISY_DATA_ANALYSIS_FILE)
    if os.path.exists(noisy_data_analysis_filepath):
        print(f"Loading existing noisy data analysis file from: {noisy_data_analysis_filepath}")
        with open(noisy_data_analysis_filepath, "r") as f:
            noisy_data_analysis_object = json.load(f)
        return noisy_data_analysis_filepath, noisy_data_analysis_object
    else: 
        print(f"Creating noisy data analysis file from: {noisy_data_analysis_filepath}")
        noisy_data_analysis_object = defaultdict(dict)
        for task_name in result.taskLanguage:
            noisy_data_analysis_object[task_name][LANGUAGE_DESCRIPTIONS_TAG] = result.taskLanguage[task_name]
            
        with open(noisy_data_analysis_filepath, "w") as f:
            json.dump(noisy_data_analysis_object, f)
        return noisy_data_analysis_filepath, noisy_data_analysis_object

def interactive_labeling_for_noise_analysis(result, noisy_data_analysis_object):
    # Label semantic incorrectness. Criterion: the language is wrong or missing information necessary to reconstruct the full task.
    print("Launching interactive labeling for noise analysis: semantic incorrectness....")
    for index, task_name in enumerate(noisy_data_analysis_object):
        try:
            descriptions = noisy_data_analysis_object[task_name][LANGUAGE_DESCRIPTIONS_TAG]
            should_label = True
            if SEMANTIC_INCORRECTNESS_TAG in noisy_data_analysis_object[task_name]:
                print(f"Found existing data for [{task_name}]: ")
                print(noisy_data_analysis_object[task_name][SEMANTIC_INCORRECTNESS_TAG])
                should_label = len(input("Hit any key to relabel: (default - no)").strip()) > 0
            if should_label:
                print(f"\nNow on: {index}/{len(noisy_data_analysis_object)}:  [{task_name}]")
                semantic_incorrectness_object = {SEMANTIC_CORRECT : [], SEMANTIC_PARTIAL_CORRECT : [], SEMANTIC_INCORRECT : []}
                for description in descriptions:
                    print(f"==>\t{description}")
                    semantic_incorrectness = int(input("Semantically correctness: (1 - no, 2 - partial, 3 - yes) ?"))
                    if semantic_incorrectness ==  3:
                        semantic_incorrectness_object[SEMANTIC_CORRECT].append(description)
                    elif semantic_incorrectness == 2:
                        semantic_incorrectness_object[SEMANTIC_PARTIAL_CORRECT].append(description)
                    else:
                        semantic_incorrectness_object[SEMANTIC_INCORRECT].append(description)
                noisy_data_analysis_object[task_name][SEMANTIC_INCORRECTNESS_TAG] = semantic_incorrectness_object
        except:
            return noisy_data_analysis_object
    return noisy_data_analysis_object
            
def run_language_data_analysis_for_noisy_data(result, tasks, testingTasks, languageDataset, languageDatasetDir):
    noisy_data_analysis_filepath = os.path.join(languageDatasetDir, languageDataset[0], NOISY_DATA_ANALYSIS_FILE)
    if os.path.exists(noisy_data_analysis_filepath):
        print(f"Loading existing noisy data analysis file from: {noisy_data_analysis_filepath}")
        with open(noisy_data_analysis_filepath, "r") as f:
            noisy_data_analysis_object = json.load(f)
    
    # Correctness metric: semantic incorrectness.
    noisy_data_analysis_object = show_semantic_incorrectness_metric(noisy_data_analysis_object)
    # Synthetic:human and human:human BLEU scores.
    noisy_data_analysis_object = show_bleu_score_metrics(noisy_data_analysis_object)

    # How do noise metrics correlate with task ground truth difficulty?
    
    # For synthetic train, human test: on the test set, did {semantic correctness; BLEU} correlate with DIFFERENIALS in solve time / not solving at all over the synthetic train, synthetic test?
    
    # For human train, human test: on the training set AND the test set, did {semantic correctness; BLEU} correlate with DIFFERENIALS in solve time / not solving at all over the synthetic train, synthetic test?
    
    print(f"Writing out final data to {noisy_data_analysis_filepath}")
    with open(noisy_data_analysis_filepath, "w") as f:
        json.dump(noisy_data_analysis_object, f)
    sys.exit(0)

def plot_noise_vs_ground_truth_hardness(train_tasks,test_tasks, noisy_data_analysis_object):
    """Plots noise metrics (semantic correctnes; BLEU score) vs. ground truth difficulty on training tasks, test tasks.
    """
    def make
    for (split, tasks) in [('train', train_tasks), ('test', test_tasks)]:
        for task in tasks:
            noisy_data_analysis_object[task.name][SPLIT_TAG] = split
            noisy_data_analysis_object[task.name][GROUND_TRUTH_LOG_LIKELIHOOD_TAG] = task.groundTruthLogLikelihood
        # Plot against semantic correctness
        ground_truth_log_likelihoods = [noisy_data_analysis_object[task.name][GROUND_TRUTH_LOG_LIKELIHOOD_TAG] for task in tasks]
        average_semantic_correctness = [noisy_data_analysis_object[task.name][SEMANTIC_INCORRECTNESS_TAG][AVERAGE_SEMANTIC_INCORRECTNESS] for task in tasks]
        
        # Plot against synthetic: human BLEU
        synthetic_human_bleu = [noisy_data_analysis_object[task.name][BLEU_SCORE_TAG][AVERAGE_SYNTHETIC_TO_HUMAN_BLEU] for task in tasks]
        
        # Plot against human: human BLEU
        interannotator_bleu = [noisy_data_analysis_object[task.name][BLEU_SCORE_TAG][AVERAGE_INTERANNOTATOR_BLEU] for task in tasks]

def show_bleu_score_metrics(noisy_data_analysis_object):
    """Calculates BLEU score with respect to the synthetic data. Mutates and appends semantic incorrectness data.
    """
    chencherry_smoothing = bleu_score.SmoothingFunction()
    
    average_synthetic_bleu = []
    average_interannotator_bleu = []
    for task_name in noisy_data_analysis_object:
        synthetic_tokens = task_name.split()
        human_descriptions_tokenized = [description.split() for description in noisy_data_analysis_object[task_name][LANGUAGE_DESCRIPTIONS_TAG]]
        
        synthetic_to_human_bleu = [bleu_score.sentence_bleu([synthetic_tokens], human_tokens, smoothing_function=chencherry_smoothing.method1, weights=(1.0, 0,0,0)) for human_tokens in human_descriptions_tokenized]
        
        human_to_human_bleu = []
        for idx_a, human_description_a in enumerate(human_descriptions_tokenized):
            for idx_b, human_description_b in enumerate(human_descriptions_tokenized):
                if idx_a != idx_b:
                    human_to_human_bleu.append(bleu_score.sentence_bleu([human_description_a], human_description_b, smoothing_function=chencherry_smoothing.method1, weights=(1.0, 0,0,0)))
        noisy_data_analysis_object[task_name][BLEU_SCORE_TAG] = {
            SYNTHETIC_TO_HUMAN_BLEU : synthetic_to_human_bleu,
            INTERANNOTATOR_BLEU : human_to_human_bleu,
            AVERAGE_SYNTHETIC_TO_HUMAN_BLEU : np.mean(synthetic_to_human_bleu),
            AVERAGE_INTERANNOTATOR_BLEU : np.mean(human_to_human_bleu)
        }
        average_synthetic_bleu.append(np.mean(synthetic_to_human_bleu))
        average_interannotator_bleu.append(np.mean(human_to_human_bleu))
    print("BLEU score metric: ")
    print(f"\t=> Interannotator Mean: {np.mean(average_interannotator_bleu)} | Median: {np.median(average_interannotator_bleu)} tasks.")
    print(f"\t=> Synthetic : Human Mean: {np.mean(average_synthetic_bleu)} | Median: {np.median(average_synthetic_bleu)} tasks.")
    return noisy_data_analysis_object

def show_semantic_incorrectness_metric(noisy_data_analysis_object):
    """Calculates semantic incorrectness. Mutates and appends semantic incorrectness data.
    """
    print("Semantic incorrectness metric: ")
    all_semantic_incorrectness = []
    for task_name in noisy_data_analysis_object:
        semantic_incorrectness_object = noisy_data_analysis_object[task_name][SEMANTIC_INCORRECTNESS_TAG]
        semantic_incorrectness_array = [0.0] * len(semantic_incorrectness_object[SEMANTIC_INCORRECT]) + [0.5] * len(semantic_incorrectness_object[SEMANTIC_PARTIAL_CORRECT]) + [1.0] * len(semantic_incorrectness_object[SEMANTIC_CORRECT])
        average_semantic_correctness = np.mean(semantic_incorrectness_array)
        semantic_incorrectness_object[AVERAGE_SEMANTIC_INCORRECTNESS] = average_semantic_correctness
        all_semantic_incorrectness.append(average_semantic_correctness)
    print(f"\t=> Found data for {len(all_semantic_incorrectness)} / {len(noisy_data_analysis_object)} tasks.")
    print(f"\t=> Mean: {np.mean(all_semantic_incorrectness)} | Median: {np.median(all_semantic_incorrectness)} tasks.")
    return noisy_data_analysis_object