
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
from tqdm import tqdm


################
# * PRINTING * #
################

# overrides the global print
def print(*args,**kwargs):
    s = ''.join(args)
    tqdm.write(s,**kwargs)
def print_if(s,cond):
    from dreamcoder.matt.sing import sing
    if sing.cfg.printif[cond]:
        print(s)
# redefined color prints to use the local print()
def green(s):
    print(mlb.mk_green(s))
def red(s):
    print(mlb.mk_red(s))
def purple(s):
    print(mlb.mk_purple(s))
def blue(s):
    print(mlb.mk_blue(s))
def cyan(s):
    print(mlb.mk_cyan(s))
def yellow(s):
    print(mlb.mk_yellow(s))
def gray(s):
    print(mlb.mk_gray(s))


def cls_name(v):
    return v.__class__.__name__

def which(cfg, no_yaml=False):
    regex = cwd_path().parent.name + '%2F' + cwd_path().name
    return f'''
{"" if no_yaml else yaml(cfg)}
cwd: {cwd_path()}
tensorboard: http://localhost:6696/#scalars&regexInput={regex}
commit: {cfg.commit}
dirty: {cfg.dirty}
argv: {cfg.argv}
start time: {cfg.start_time}
curr time: {timestamp()}
    '''.strip()

def yaml(cfg):
    return OmegaConf.to_yaml(cfg)

def timestamp():
    return datetime.now()

#############
# * PATHS * #
#############

"""
Please read these points if you want to know how to make a certain path.
- the reason all these path things are functions is because chdir can change them. Or at least all the cwd ones
- everything (including cwd_path()) is absolute paths, see relative paths guide below if you want otherwise
- relative paths guide:
    if you want: /scratch/mlbowers/proj/example_project/outputs/12-31-20/12-23-23/x.png -> x.png
        * then do: p.relative_to(cwd_path())
    if you want: /scratch/mlbowers/proj/example_project/outputs/12-31-20/12-23-23/x.png -> 12-31-20/12-23-23/x.png
        * then do: p.relative_to(outputs_path())
- If you want .../DATE/TIME path just use cwd_path()
    - if you want DATE/TIME without that relpath bit just do cwd_path().relative_to(outputs_path())

"""

def init_paths():
    assert cwd_path() != toplevel_path(), "must be called when already within an experiment folder"
    saves_path().mkdir()
    plots_path().mkdir()

def toplevel_path():
    """
    Same as the overall git repo path.
    /scratch/mlbowers/proj/ec/
    """
    return Path(hydra.utils.to_absolute_path(''))

def outputs_path():
    """
    The path to the 'outputs' directory
    Out: /scratch/mlbowers/proj/ec/outputs
    """
    return toplevel_path() / 'outputs'

def testgen_path():
    """
    Where tests get read from and written to by mode=testgen
    /scratch/mlbowers/proj/ec/testgen
    """
    return toplevel_path() / 'testgen'

def cwd_path():
    """
    Out: /scratch/mlbowers/proj/ec/output/12-31-20/12-23-23
    """
    return Path(os.getcwd())

def saves_path():
    """
    the saves folder
    /scratch/mlbowers/proj/example_project/outputs/12-31-20/12-23-23/saves
    """
    return cwd_path() / 'saves'

def plots_path():
    """
    the plots folder
    /scratch/mlbowers/proj/example_project/outputs/12-31-20/12-23-23/plots
    """
    return cwd_path() / 'plots'

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
            res.extend(list(outputs_path().glob(r)))
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

def uncurry(fn,args):
    """
    if youd normally call the fn like fn(a)(b)(c)
    you can call it like uncurry(fn,[a,b,c])
    """
    if len(args) == 0:
        return fn()
    res = None
    for arg in args:
        if res is not None:
            res = res(arg)
        else:
            res = fn(arg)
    return res