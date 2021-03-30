
from dreamcoder.matt.util import *
import torch
from dreamcoder.matt.sing import sing
from dreamcoder import loader
dill

def main():

    testgen_path().mkdir(exist_ok=True)

    assert sing.cfg.mode == 'testgen'
    tg = sing.cfg.testgen
    if tg.from_fn is None:
        die('missing argument testgen.from_fn')
    if tg.to_file is None:
        die('missing argument testgen.to_file')
    if tg.num_tasks is None:
        die('missing argument testgen.num_tasks')

    outfile = testgen_path() / tg.to_file
    
    try:
        testgen_fn = eval(tg.from_fn)
    except Exception as e:
        die(f"cant find fn `{tg.from_fn}` (Actual exception: {e})")
    
    assert callable(testgen_fn)

    frontiers = testgen_fn(sing.cfg)
    st = SavedTest(sing.cfg,frontiers)
    st.save(outfile)
    

class SavedTest:
  def __init__(self,cfg,fs) -> None:
    self.cfg = cfg
    self.fs = fs
  def save(self,name):
    path = with_ext(testgen_path() / name, 'tgen')
    if move_existing(path):
        red("WARNING: MOVED EXISTING TESTGEN FILE")
    torch.save(self,path, pickle_module=dill)
    print(f"saved testgen with name {path}")


def deepcoder(cfg):
    taskloader = loader.DeepcoderTaskloader(test=True)

    tasks = taskloader.test_tasks(n=cfg.testgen.num_tasks)

    assert len(tasks) == cfg.testgen.num_tasks
    return tasks

def joshTasks(w):
    """
    From https://github.com/ellisk42/ec/blob/Josh/dreamcoder/domains/list/makeListTasks.py
    """
    ts = []
    import json
    if w == "1":
        directory = "data/wave1"
    elif w == "2":
        directory = "data/wave2"
    elif w == "3":
        directory = "data/wave3/json"
    elif w == "3.1":
        directory = "data/wave3.1/json"
    elif w == "final":
        directory = "data/final_wave"
    else:
        assert False
    directory = utils.to_absolute_path(directory)
    for fn in os.listdir(directory):
        if not fn.endswith(".json"):continue

        if w == "final":
            if not (fn.endswith("_1.json")):
                continue

        with open(f"{directory}/{fn}") as handle:
            data = json.load(handle)

            ts.append(Task(data.get("name",fn.split(".")[0][1:]),
                           arrow(tlist(tint),tlist(tint)),
                           [((e["i"],),e["o"])
                            for e in data["data"] ]))
    return list(sorted(ts,key=lambda t: t.name))

def josh(cfg):
    tasks = joshTasks(str(cfg.test.josh.wave))
    frontiers = [FakeFrontier(None,task) for task in tasks]
    return frontiers

def lucas(cfg):
    from dreamcoder.domains.list.main import retrieveJSONTasks, sortBootstrap, make_list_bootstrap_tasks
    def get_tasks(f):
        return retrieveJSONTasks(utils.to_absolute_path(f))
    if cfg.test.lucas.version == 1:
        tasks = get_tasks("data/list_tasks2.json")[:105]
    elif cfg.test.lucas.version == 2:
        tasks = get_tasks("data/list_tasks2.json")[:4928]
    elif cfg.test.lucas.version == 3:
        tasks = get_tasks("data/list_tasks2.json")
    elif cfg.test.lucas.version == 'old':
        tasks = get_tasks("data/list_tasks.json") + sortBootstrap()
    elif cfg.test.lucas.version == 'boopstrap':
        tasks = make_list_bootstrap_tasks()
    else:
        raise ValueError
    frontiers = [FakeFrontier(None,task) for task in tasks]
    return frontiers