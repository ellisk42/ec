def getCleanLanguage(language_data, clean_vocab):
    clean_data = []
    for s in language_data:
        tokens = s.split()
        clean = [t for t in tokens if t in clean_vocab]
        clean = " ".join(clean)
        clean = clean.strip()
        if len(clean) > 0:
            clean_data.append(clean)
    return clean_data
    
def languageForTasks(languageDataset, languageDatasetDir, taskDict, clean_vocab=None):
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
    
    if not clean_vocab:
        vocabularies = {"train": set(), "test": set()}
    else:
        vocabularies = clean_vocab
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
                            if clean_vocab:
                                taskDict[t_name] = getCleanLanguage(languageData[t_name], vocabularies[split])
                                if split == "test":
                                    print(f"Task: {t_name}")
                                    clean = '\n'.join(taskDict[t_name])
                                    print(f"Clean data: {clean}")
                                    print("\n")
                            else:
                                taskDict[t_name] = languageData[t_name]
                if not clean_vocab:
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
        
    