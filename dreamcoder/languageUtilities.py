def languageForTasks(languageDataset, taskDict):
    """
    Loads a languageDataset : {task_paths : list of [sentence, other_data] tuples}
    :ret: {t : list of sentences for each task.}
    """
    import json
    import os.path as osp
    
    with open(languageDataset, 'r') as f:
        languageDataset = json.load(f)
    
    def extract_task_name(task_path):
        return osp.basename(task_path).split('name_')[-1].split('.')[0]
    languageDataset = {
        extract_task_name(task_path) : [data[0] for data in language_data]
        for (task_path, language_data) in languageDataset.items()
    }
    
    def task_to_safe_name(task_name): 
        return task_name.replace("=","_").replace(' ','_').replace('/','_').replace("-","_").replace(")","_").replace("(","_")
    
    # Check for collisions in safe name.
    from collections import defaultdict
    task_safe_names = defaultdict(list)
    for t in taskDict:
        task_safe_names[task_to_safe_name(t.name)].append(t.name)
        
    languageForTasks = {}
    for t in taskDict:
        task_safe_name = task_to_safe_name(t.name)
        if task_safe_name not in languageDataset or len(languageDataset[task_safe_name]) < 1:
            print("Missing language for : {}".format(t.name))
        if len(task_safe_names[task_safe_name]) > 1:
            print("Collision: safe-name: {}, names: {}".format(task_safe_name, task_safe_name[task_safe_name]))
        language_for_task = languageDataset[task_safe_name] if task_safe_name in languageDataset else []
        languageForTasks[t] = language_for_task
    print("Found language for {}/{} tasks".format(len([t for t in languageForTasks if len(languageForTasks[t]) > 0]), len(languageForTasks)))
    return languageForTasks
        
    