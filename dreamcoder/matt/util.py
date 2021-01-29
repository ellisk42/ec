
import sys,os
import mlb

from collections import OrderedDict
import hydra
import itertools
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
    s = ''.join([str(arg) for arg in args])
    tqdm.write(s,**kwargs)
def die(*args,**kwargs):
    print(*args,**kwargs)
    sys.exit(1)
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


def with_ext(path,ext):
    if not str(path).endswith(f'.{ext}'):
        return Path(f'{path}.{ext}')
    return path


def get_unique_path(orig_path):
    """
    Takes a path like foo/bar/baz.xyz and returns a path like foo/bar/baz.old0.xyz
    or however much naming is needed to avoid conflict.
    """
    p = Path(orig_path) # in case its not
    i=0
    while p.exists():
        if orig_path.suffix != '': # has file extension
            p = pathlib.Path(f'{orig_path.stem}.old{i}.{orig_path.suffix}')
        else:
            p = pathlib.Path(f'{orig_path}.old{i}')
        i += 1
    return p


def move_existing(path):
    """
    Does nothing if `path` doesnt exist.
    If `path` does exist then rename it eg from foo/bar/baz.xyz to foo/bar/baz.old0.xyz

    Mainly used in cases where we usually do want to overwrite the file but we want to keep the old one around just in case
    """
    if path.exists(): # super rare honestly bc `job` includes timestamps in names
        safe_name = get_unique_path(path)
        path.rename(safe_name)
        red(f"moved file: {path.relative_to(outputs_path())} -> {safe_name.relative_to(outputs_path())}")
        return True
    return False


def toplevel_plots_path():
    """
    Same as the overall git repo path.
    /scratch/mlbowers/proj/ec/outputs/_toplevel
    """
    return outputs_path() / '_toplevel'

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

def model_results_path():
    """
    the saves folder
    /scratch/mlbowers/proj/example_project/outputs/12-31-20/12-23-23/model_results
    """
    return cwd_path() / 'model_results'

def plots_path():
    """
    the plots folder
    /scratch/mlbowers/proj/example_project/outputs/12-31-20/12-23-23/plots
    """
    return cwd_path() / 'plots'

def outputs_search(regexes, sort=True, ext=None):
    """
    The union of one or more regexes treated like `outputs/**/*{regex}`
        * regexes :: str | [str]
        * duplicates are removed
        * note theres no '*' built into the end of the regex so you should add one yourself if you want one
        * sorts results unless sort=False

    Returns a list of Paths
    """
    if isinstance(regexes,str):
        regexes = [regexes]

    # glob all the paths, dedup using OrderedDict.fromkeys() (like set() but preserves order)
    paths = list(OrderedDict.fromkeys(itertools.chain.from_iterable([list(outputs_path().glob(f'**/*{regex}')) for regex in regexes])))
    if sort:
        paths = sorted(paths)
    if ext:
        paths = [p for p in paths if p.suffix == ext]
    return paths


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