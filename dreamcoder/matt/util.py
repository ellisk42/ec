
import sys,os
import mlb

import hydra
from omegaconf import DictConfig,OmegaConf,open_dict
from datetime import datetime
import pathlib
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import contextlib
from time import time

def cls_name(v):
    return v.__class__.__name__

def which(cfg):
    print(yaml(cfg))
    print(getcwd())
    regex = getcwd().parent.name + '%2F' + getcwd().name
    print(f'http://localhost:6696/#scalars&regexInput={regex}')
    print("curr time:",timestamp())

def getcwd():
    return Path(os.getcwd())

def yaml(cfg):
    return OmegaConf.to_yaml(cfg)

def timestamp():
    return datetime.now()

## PATHS


def toplevel_path():
    """
    /scratch/mlbowers/proj/example_project/
    """
    return Path(hydra.utils.to_absolute_path(''))

def outputs_path(p):
    """
    Out: /scratch/mlbowers/proj/example_project/outputs/12-31-20/12-23-23
    """
    return toplevel_path() / 'outputs/'

def saves_path():
    """
    /scratch/mlbowers/proj/example_project/outputs/12-31-20/12-23-23/saves
    """
    return toplevel_path() / 'saves'

def toplevel_path(p):
    """
    In:  plots/x.png
    Out: /scratch/mlbowers/proj/example_project/plots/x.png
    """
    return Path(hydra.utils.to_absolute_path(p))

def outputs_path(p):
    """
    In:  plots/x.png
    Out: /scratch/mlbowers/proj/example_project/outputs/12-31-20/12-23-23/plots/x.png
    """
    return toplevel_path('outputs') / p

def outputs_relpath(p):
    """
    In:  plots/x.png
    Out: 12-31-20/12-23-23/plots/x.png
    """
    return outputs_path(p).relative_to(outputs_path(''))

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
            r = f'**/*{r}'
            res.extend(list(outputs_path('').glob(r)))
        except ValueError as e:
            print(e)
            return []
    return sorted(res)


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

def unthread():
    """
    disables parallelization
    """
    import os
    if 'numpy' in sys.modules:
        mlb.yellow("warning: unthread() might not work properly if done after importing numpy")
    
    os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
    os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
    import torch
    torch.set_num_threads(1)

def deterministic(seed):
    import torch
    import numpy as np
    import random
    torch.manual_seed(seed)
    # warning: these may slow down your model
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

class NoPickle: pass
class Saveable:
  """
  Things you can do:
    * define a class variable no_save = ('hey','there') if you want self.hey and self.there to not be pickled (itll be replaced with the sentinel NoPickle)
    * define a method `post_load(self) -> None` which will be called by setstate after unpickling. Feel free to call it yourself in __init as well if you want it during __init too.
  Extras:
    * a repr() implementation is written for you
    * __getitem and __setitem are defined for you so you can do self.k instead of self.__dict__['k']. Note that it errors if you try to overwrite an @property
    * .update(dict) is defined for you and does bulk setattr

  """
  no_save = () # tuple so it's not mutably shared among instances
  def __getstate__(self):
    return {k:(v if k not in self.no_save else NoPickle) for k,v in self.__dict__.items()}
  def __setstate__(self,state):
    self.__dict__.update(state)
    self.post_load()
  def post_load(self):
	  pass
  def __getitem__(self,key):
      return getattr(self,key)
  def __setitem__(self,key,val):
    if hasattr(self,key) and isinstance(getattr(type(self),key), property):
      raise ValueError("Trying to overwrite an @property")
    return setattr(self,key,val)
  def __repr__(self):
      body = []
      for k,v in self.__dict__.items():
          body.append(f'{k}: {repr(v)}')
      body = '\n\t'.join(body)
      return f"{self.__class__.__name__}(\n\t{body}\n)"
  def update(self,dict):
      for k,v in dict.items():
          self[k] = v
  def clone_from(self,other):
      for k,v in other.__dict__.items():
          self[k] = v

class RunningFloat:
    def __init__(self):
        self.reset()
    def add(self,x):
        self.vals.append(float(x))
    def count(self):
        return len(self.vals)
    def avg(self):
        if self.count() == 0:
            return 0
        return sum(self.vals)/len(self.vals)
    def reset(self):
        self.vals = []
        self.tstart = time()
    def elapsed(self):
        return time() - self.tstart
    def rate(self):
        return self.count() / self.elapsed()
