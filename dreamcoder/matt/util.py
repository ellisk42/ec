import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig,OmegaConf,open_dict
from datetime import datetime
import os
import pathlib

def ec_path(p):
    return pathlib.Path(to_absolute_path(p))
def outputs_path(p):
    return ec_path('outputs') / p
def toplevel_path(p):
    return outputs_path('_toplevel') / p
def printable_local_path(p):
    """
    In:  plots/x.png
    Out: 12-31-20/12-23-23/plots/x.png
    """
    return hide_path_prefix(pathlib.Path(os.getcwd())) / p
def yaml(cfg):
    print(OmegaConf.to_yaml(cfg))
def timestamp():
    return datetime.now()
def which(cfg):
    if cfg is None:
        return
    yaml(cfg)
    print(os.getcwd())
    regex = os.path.basename(os.path.dirname(os.getcwd())) + '%2F' + os.path.basename(os.getcwd())
    print(f'http://localhost:6696/#scalars&regexInput={regex}')
    print("curr time:",timestamp())

def outputs_regex(*rs):
    """
    The union of one or more regexes over the outputs/ directory.
    Returns a list of results (pathlib.Path objects)
    """
    res = []
    for r in rs:
        r = r.strip()
        if r == '':
            continue # use "*" instead for this case please. I want to filter out '' bc its easy to accidentally include it in a generated list of regexes
        try:
            res.extend(list(outputs_path('').glob(f'**/*{r}*')))
        except ValueError as e:
            print(e)
            return []
    return sorted(res)

def hide_path_prefix(p):
    """
    remove everything in path before the `outputs` dir (inclusive) eg:
    In:  /scratch/mlbowers/proj/ec/outputs/2020-09-17/15-22-30
    Out: 2020-09-17/15-22-30
    """
    idx = p.parts.index('outputs')+1 # points to DATE dir
    return pathlib.Path(*p.parts[idx:])

def get_datetime_path(p):
    """
    Path -> Path
    In:  .../2020-09-14/23-31-49/t3_reverse.no_ablations_first
    Out: .../2020-09-14/23-31-49
    Harmless on shorter paths
    """
    idx = p.parts.index('outputs')+3 # points one beyond TIME dir
    return pathlib.Path(*p.parts[:idx]) # only .../DATE/TIME dir
def get_datetime_paths(paths):
    return [get_datetime_path(p) for p in paths]

def filter_paths(paths, predicate):
    return [p for p in paths if predicate(p)]

    # then filter using predicates
    for predicate in [arg for arg in args if '=' in arg]:
        lhs,rhs = predicate.split('=')
        # TODO WAIT FIRST JUST FOLLOW THIS https://github.com/tensorflow/tensorboard/issues/785
        # idk it might be better.
        # TODO first navigate to the actual folder that the tb files are in bc thats 
        # what process() should take as input (e.g. 'tb' or whatever prefix+name is)
        process(result)
        raise NotImplementedError

    return results
