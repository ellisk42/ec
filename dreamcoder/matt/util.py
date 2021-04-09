
import sys,os
import mlb

from collections import OrderedDict
import hydra
import itertools
from omegaconf import DictConfig,OmegaConf,open_dict
from datetime import datetime
import pathlib
from pathlib import Path
import heapq
from torch.utils.tensorboard import SummaryWriter
import contextlib
import math
import random
import numpy as np
import time
from tqdm import tqdm
import inspect
import torch
import omegaconf

from dreamcoder.matt.sing import sing



# making these public to `from util import *` modules!
from einops import rearrange, reduce, repeat
from torch import cat,stack

def flatten(list_of_lists):
    return itertools.chain.from_iterable(list_of_lists)

def group_by(iter, key):
  res = defaultdict(list)
  for v in iter:
    res[key(v)].append(v)
  return res

def pad_list_list_tensor(list_list_tensor):
    """
    takes a list of list of same-dimensionality tensors and pads them all to be
    equal in inner list length so that they can all be stacked. Pads with zero tensors.
    No sequences can have zero length.
    Actually works with multidimensional inner tensors too


    list_list_tensor :: [BATCH, RAGGED_SEQ, H] where BATCH and RAGGED_SEQ are list dimensions and RAGGED_SEQ is ragged (varies between instances)
    returns res,mask where
        res :: [BATCH,MAX_SEQ,H] where MAX_SEQ is the maximum sequence length in among all the RAGGED_SEQs
        mask :: [BATCH,MAX_SEQ] is a booltensor useful as a key_padding_mask or attn_mask with 0 at padding locations and 1 elsewhere

    """
    longest = max(len(l) for l in list_list_tensor)
    H = list_list_tensor[0][0].shape

    mask = torch.zeros(len(list_list_tensor), longest,     device=sing.device, dtype=bool)
    res =  torch.zeros(len(list_list_tensor), longest, *H, device=sing.device, dtype=bool)
    for list_tensor,submask,subres in zip(list_list_tensor,mask,res):
        submask[len(list_tensor):] = True
        subres[len(list_tensor):] = list_tensor
    
    return res,mask

class InvalidSketchError(Exception): pass

################
# * PRINTING * #
################

# overrides the global print
def print(*args,**kwargs):
    s = ''.join([str(arg) for arg in args])
    tqdm.write(s,**kwargs)


old_tensor_repr = torch.Tensor.__repr__
def tensor_repr(tensor):
    old_repr = old_tensor_repr(tensor)
    return f'T{list(tensor.shape)} {old_repr}'
torch.Tensor.__repr__ = tensor_repr

def die(s):
    red(s)
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

def count_frames(name):
    """
    return number of frames in call stack (relative to
    this count_frames() function ie including the caller) corresponding
    to a function call with the given function name.

    * THIS FUNCTION IS EXTREMELY SLOW

    """
    call_stack_fn_names = [x.function for x in inspect.stack()]
    return call_stack_fn_names.count(name)

class AttrDict(dict):
    def __getattr__(self,k):
        if k in self:
            return self[k]
        raise AttributeError(k)
    def __setattr__(self, k, v):
        if k not in self and not isinstance(v,AttrDictOverride):
            raise Exception(f"cant add new field `{k}` to {self}")
        if isinstance(v,AttrDictOverride):
            v = v.val
        self[k] = v
class AttrDictOverride:
    def __init__(self,val):
        self.val=val

def omeconf_to_attrdict(cfg):
    if not isinstance(cfg,omegaconf.dictconfig.DictConfig):
        return cfg
    return AttrDict({k:omeconf_to_attrdict(v) for k,v in cfg.items()})
def attrdict_to_dict(cfg):
    if not isinstance(cfg,AttrDict):
        return cfg
    return {k:attrdict_to_dict(v) for k,v in cfg.items()}


class DepthPrinter:
    def __init__(self,on=True) -> None:
        self.depth = 0
        self.on = on
    def __call__(self,*args,indent=False,dedent=False,**kwargs):
        if self.on is False:
            return
        if indent:
            self.indent()
        print('  '*self.depth,*args,**kwargs)
        if dedent:
            self.dedent()
    def indent(self):
        self.depth += 1
    def dedent(self):
        self.depth -= 1
    def reset(self):
        self.depth = 0

def short_repr(obj):
    if torch.is_tensor(obj):
        return f'Tensor({obj.shape})'
    return repr(obj)


def compressed_str(s):
    return s.replace('\n',' ').replace('\t',' ').replace(' ','')

def cls_name(v):
    return v.__class__.__name__

def which(cfg, no_yaml=False):
    regex = cwd_path().parent.name + '%2F' + cwd_path().name
    yaml_str = "" if no_yaml else yaml(cfg) + "\n\n"
    return f'''
{yaml_str}
full_name: {cfg.full_name}
cwd: {cwd_path()}
tensorboard: http://localhost:6696/#scalars&regexInput={regex}
commit: {cfg.commit}
dirty: {cfg.dirty}
argv: {cfg.argv}
start time: {cfg.start_time}
curr time: {timestamp()}
    '''.strip()

def yaml(cfg):
    return OmegaConf.to_yaml(OmegaConf.create(attrdict_to_dict(cfg)))

def timestamp():
    return datetime.now()
def timestamp_to_filename(dt):
    return dt.strftime(f'%m-%d.%H-%M-%S')

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
    model_results_path().mkdir()

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
        p = orig_path.with_suffix(f'.old_{i}{orig_path.suffix}') # eg foo.tmp -> foo.old_0.tmp or foo -> foo.old_0
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
        red(f"moved file: {path} -> {safe_name}")
        return True
    return False

def toplevel_plots_path():
    """
    Same as the overall git repo path.
    /scratch/mlbowers/proj/ec/outputs/_toplevel
    """
    return outputs_path() / 'toplevel'

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

def train_path():
    """
    The path to the 'outputs' directory
    Out: /scratch/mlbowers/proj/ec/outputs
    """
    return outputs_path() / 'train'

def test_path():
    """
    The path to the 'outputs' directory
    Out: /scratch/mlbowers/proj/ec/outputs
    """
    return outputs_path() / 'test'

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


def is_datedir(p):
    """
    foo/bar/2021-02-09 -> True (date dir)
    foo/bar/2021-02-09/12-23-12 -> False (time dir)
    foo/bar -> False (other dir)
    foo/bar/2021-02-09/x.py -> False (not a dir)
    """
    p = p.name
    template = "2021-02-09"
    if len(p) != len(template):
        return False
    for c,t in zip(p,template):
        if t.isdigit():
            if not c.isdigit():
                return False # both shd be digits
        else:
            if c != t:
                return False # if template is a non-digit (ie a dash), they should match exactly
    return True


def path_search(top_dir, regexes, sort=True, ext=None, expand=False, rundirs=False):
    """
    Returns a list of Paths
    The union of one or more regexes treated like `outputs/**/*{regex}`
        * regexes :: str | [str]
        * duplicates are removed
        * note theres no '*' built into the end of the regex so you should add one yourself if you want one
        * sorts results unless sort=False
        * if `ext` is True then filter out any files that dont have the proper extension
            Often used with `expand`
        * if `expand` is True then convert every matched directory to all if its children and their children etc (**/*)
            Explodes size of output but may be desirable for grabbing exact files
        * if `rundirs` is True then reduce each path to its DATE/TIME folder, removing duplicates
            Often useful if you want the DATE/TIME dirs based on matching a regex on their contents
    """
    top_dir = Path(top_dir)
    assert top_dir.is_dir()

    if isinstance(regexes,str):
        regexes = [regexes]

    
    # glob all the paths, dedup using OrderedDict.fromkeys() (like set() but preserves order)
    paths = list(OrderedDict.fromkeys(itertools.chain.from_iterable([list(top_dir.glob(f'**/*{regex}')) for regex in regexes])))
    assert not any(len(p.parts) == 0 for p in paths), "you seem to have regexed the entire directory"

    if expand:
        raise NotImplementedError
        paths = itertools.chain.from_iterable([(p.glob('**/*') if p.is_dir() else p) for p in paths])
    if ext:
        if not ext.startswith('.'):
            ext = '.' + ext
        paths = [p for p in paths if p.suffix == ext]
    if rundirs:
        for p in paths:
            if is_datedir(p):
                paths.extend(p.glob('*')) # expand a lone DATE dir to all its DATE/TIME children
        paths = [p for p in paths if not is_datedir(p)] # throw out those lone DATE dirs you expanded
        paths = [get_rundir(p) for p in paths] # convert children to their parent rundir
        paths = list(OrderedDict.fromkeys(paths)) # dedup while maintaining order
    if sort:
        paths = sorted(paths)
    return paths


def get_rundir(p):
    """
    Given  foo/bar/outputs/MODE/DATE/TIME.job.run/x/y
    Return foo/bar/outputs/MODE/DATE/TIME.job.run/

    If orig_path was valid to the caller, then the return value here
    will be valid to the caller (whereas the simple path MODE/DATE/TIME would only be valid if the caller were in the
    outputs directory as their cwd)
    """
    orig_path = p
    while not is_datedir(p.parent): # if our parent is a datedir, clearly we are the rundir
        if p == p.parent: # at root of path
            raise ValueError(f"path {orig_path} doesnt contain a rundir (can't find the datedir that should be right above it)")
        p = p.parent
    
    assert p.parent.parent.parent.name == outputs_path().name, "sanity check"
    return p


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


def cfg_get(cfg, keypath):
    """
    Given a cfg and something like keypath=`pnode.model.whatever` this accesses that field and returns its value
    """
    zipper = keypath.split('.') # 'model.pnode.whatever' -> ['model','pnode','whatever']
    inner = cfg
    for key in zipper:
        inner = inner[key]
    return inner

def cfg_set(cfg, keypath, val):
    """
    Given a cfg and something like keypath=`pnode.model.whatever` this accesses that field and sets its value to `val`
    """
    zipper = keypath.split('.') # 'model.pnode.whatever' -> ['model','pnode','whatever']
    inner = cfg
    for key in zipper[:-1]:
        inner = inner[key]
    inner[zipper[-1]] = val

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
        self.tstart = time.time()
    def elapsed(self):
        return time.time() - self.tstart
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
        try:
            if res is not None:
                res = res(arg)
            else:
                res = fn(arg)
        except (ZeroDivisionError,FloatingPointError) as e:
            raise InvalidSketchError(e)
    return res

def lse(xs):
    """
    LogSumExp: returns log(sum([exp(x) for x in xs])) but numerically stable.
    If `lse([x-lse(xs) for x in xs])` is appx 0 you know this works right
    Used to normalize a distribution `lls` as in normalize_log_dist()
    """
    assert isinstance(xs,(list,tuple))
    largest = max(xs)
    return largest + math.log(sum([math.exp(x-largest) for x in xs]))

def normalize_nonlog_dist(xs):
    assert isinstance(xs,(list,tuple))
    total = sum(xs)
    return [x/total for x in xs]

def normalize_log_dist(xs):
    assert isinstance(xs,(list,tuple))
    total = lse(xs)
    return [x-total for x in xs]

def sample_nonlog_dist(xs):
    assert isinstance(xs,(list,tuple))
    if not math.isclose(sum(xs),1.):
        # we throw an error instead of correcting it for them in case they didnt intend for this
        raise ValueError("Please normalize distribution with normalize_nonlog_dist() or normalize_log_dist() first")
    r = random.random()
    xs = np.cumsum(xs)
    xs[-1] = 1. # bc we can't count on floats to truly sum to 1.0
    for i,x in enumerate(xs):
        if x >= r:
          return i
    assert False

def sample_log_dist(xs):
    """
    takes a list of normalized lls (this gets verified) and samples one, returning its index.
    """
    assert isinstance(xs,(list,tuple))
    xs = [math.exp(x) for x in xs]
    return sample_nonlog_dist(xs) # normalization check gets done in here (as opposed to checking lse() outside)


# HEAP
class Heap:
    def __init__(self, max_size=None, reset_to_size=None):
        """
        A min heap! Optionally provide max_size and reset_to_size. These are used as a heap with a capacity.
            * This is a min-heap (as the method pop_min() suggests)
            * When `max_size` is reached for the heap, it will cut its size down to `reset_to_size`, throwing out large elements
                This is a slightly expensive operation which is why we make reset_to_size a bit lower than
                max_size. It's not too expensive, but we'd rather not call it nonstop hence the `reset_to_size` thing
            * the .q field is a list, and if you sort() it you'll still maintain the heap invariant (yay!) so thats harmless
                This is pretty great. If you want the n cheapest elements just do sorted(heap.q)[:n]. Likewise with most expensive
                just do [:-n] after sorting. Also .sort() wont hurt it so for popping many at a time just sort it, slice it, and modify
                .q directly! For the record heap[0] is always the pop_min result.
        """
        if max_size is not None:
            assert reset_to_size is not None
            assert reset_to_size <= max_size

        self.q = []
        self.max_size = max_size
        self.reset_to_size = reset_to_size

    def push(self,x):
        heapq.heappush(self.q,x)
        if self.max_size is not None:
            if len(self.q) > self.max_size:
                # cut down to self.reset_to_size
                self.q.sort() # this actually maintains the heap invariant so it's safe to do! (https://docs.python.org/3/library/heapq.html)
                self.q = self.q[:self.reset_to_size] # cut out the final elements (most expensive) leaving the cheap ones. self.q[0] is the result of pop_min() always for example
    def pop_min(self):
        return heapq.heappop(self.q)
    def peek_min(self):
        return min(self.q)
    def __len__(self):
        return len(self.q)
    def __repr__(self):
        return f'{len(self)}: {self.q}'
