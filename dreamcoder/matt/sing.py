
"""
`sing` as in the singleton design pattern. See this page on "How do I share global variables across modules?" from the python docs:
  https://docs.python.org/3/faq/programming.html#id11

Nothing in this file is initialized because we want it to be set manually by whatever loading or init code gets run.
"""
from dreamcoder.matt.util import *
import torch.nn as nn
import torch
import traceback
import functools
from torch.utils.tensorboard import SummaryWriter
import shutil
from mlb.mail import email_me,text_me
from dreamcoder.matt import fix

class Sing(Saveable):
  no_save = ('w',)
  def __init__(self) -> None:
    pass
  def from_cfg(self, cfg):
    """
    note that this MUST modify sing inplace, it cant be a staticmethod
    bc otherwise when other modules do `from matt.sing import sing` theyll
    get a different copy if we ever reassign what sing.sing is. So it must be inplace.
    (i tested this).
    """

    if cfg.mode in ('train','profile','inspect','test'):
      if not cfg.load:
        ###########################################
        ### * MAKE NEW SING INCLUDING A MODEL * ###
        ###########################################

        if cfg.mode == 'test':
          die("can't do mode=test without a file to load from")

        self.cfg = cfg

        """
        - sing.py gets used by everyone, so for simplicity we make sing do the imports within its own functions
        - for the record, we consider util.py to be used even more so it has to import sing within its own functions
        """
        from dreamcoder.matt.train import TrainState
        from dreamcoder import models,loader

        self.train_state = TrainState(cfg)
        self.cwd = cwd_path()
        init_paths()
        self.name = self.cfg.full_name
        self.device = torch.device(cfg.device)
        self.stats = Stats()

        self.taskloader = {
          'deepcoder':loader.DeepcoderTaskloader
        }[self.cfg.data.type](train=True,valid=True)

        self.g = self.taskloader.g

        self.model = {
          'mbas': models.MBAS,
          'dc': models.Deepcoder, # todo
          'rb': models.Robustfill, # todo
        }[cfg.model.type]()

        self.model.to(self.device)
        self.set_tensorboard()
      else:
        ############################################
        ### * LOAD OLD SING CONTAINING A MODEL * ###
        ############################################

        overrided = [arg.split('=')[0] for arg in sys.argv[1:]]
        new_device = torch.device(cfg.device) if 'device' in overrided else None

        paths = path_search(train_path(),cfg.load)
        if len(paths) == 0:
            die(f'Error: cfg.load={cfg.load} doesnt match any files')
        if len(paths) > 1:
            red(f'Error: cfg.load={cfg.load} matches more than one file')
            red(f'Matched files:')
            for p in paths:
                red(f'\t{p}')
            sys.exit(1)
        [path] = paths
        if not path.suffix == '.sing': # if not a .sing file, get the parent dir then load the appropriate autosave
          rundir = get_rundir(path) # get DATE/TIME dir
          saves = list((rundir / 'saves').glob('*autosave*.sing'))
          if len(saves) == 0:
            die(f'saves folder seems to be empty: {rundir / "saves"}')
          def savenum(save):  # convert 'autosave_0000100.sing' -> 100
            stem = save.stem # strip suffix of .sing so 'autosave_0000100.sing' -> 'autosave_0000100'
            num = stem.split('_')[-1] # 'autosave_0000100' -> '0000100'
            return int(num)
          path = max(saves, key=lambda save: savenum(save))

        green(f'loading from {path}')

        _sing = torch.load(path,map_location=new_device) # `None` means stay on original device
        self.clone_from(_sing) # will even copy over stuff like SummaryWriter object so no need for post_load()
        del _sing
        if new_device is not None:
          self.device = new_device # override sings device indicator used in various places

        fix.fix_cfg(self.cfg) # any forwards compatability fixes to the loaded cfg

        if cfg.mode != 'test': # if mode == test then we actually dont apply any overrides besides the `self.device` one
          self.apply_train_overrides(overrided,cfg)

          # turn commit, is_dirty, and argv into lists and append our new values onto the old ones
          if not isinstance(cfg.commit,list):
            cfg.commit = [cfg.commit]
            cfg.is_dirty = [cfg.is_dirty]
            cfg.argv = [cfg.argv]
          self.cfg.commit = (*cfg.commit, self.cfg.commit)
          self.cfg.is_dirty = (*cfg.is_dirty, self.cfg.is_dirty)
          self.cfg.argv = (*cfg.argv, self.cfg.argv)

        print(f"chdir to {self.cwd}")
        os.chdir(self.cwd)
        self.set_tensorboard()
    elif cfg.mode in ('plot','testgen','cmd'):
      ######################################################################
      ### * BARE BONES SING: NO TRAIN_STATE, NO MODEL, NO LOADERS, ETC * ###
      ######################################################################
      self.cfg = cfg
    else:
      raise ValueError(f"mode={cfg.mode} is not a valid mode")

  def apply_train_overrides(self,overrided,cfg):
    """
    Used when loading a model and overriding some aspects of it
    Takes:
     * overrided = [arg.split('=')[0] for arg in sys.argv[1:]]
     * cfg: new config where we'll pull the values for the overrides
    If a key shows up in `overrided` and its value (cfg[key]) differs
      from `self.cfg[key]` and it is whitelisted, then we override it.
    This allows for custom behavior depending on what the override key is,
      and the whitelist setup means we wont let someone do something like
      believe theyre modifying the training data when really that gets set
      in stone when you first create a new run
    """
    whitelisted_keypaths = ('device', 'load', 'dirty', 'mode', 'check_overrides')
    whitelisted_keypaths_startswith = ()
    for keypath in overrided:
      if cfg_get(cfg,keypath) == cfg_get(self.cfg,keypath):
        continue # if theyre already equal then no worries
      if keypath not in whitelisted_keypaths and not any(keypath.startswith(start) for start in whitelisted_keypaths_startswith):
        red(f'keypath `{keypath}` is not in whitelisted overrides. Modify in sing.py:Sing.apply_overrides()')
        if cfg.check_overrides:
          sys.exit(1)
      """
      If you want any custom behavior for an override, put it here
      """
      val = cfg_get(cfg,keypath)
      cfg_set(self.cfg,keypath,val) # override it
   
  def save(self, name):
      path = with_ext(saves_path() / name, 'sing')
      print(f"saving Sing to {path}...")
      torch.save(self, f'{path}.tmp')
      shutil.move(f'{path}.tmp',path)
      print("done")
  def post_load(self):
    """
    we should not set_tensorboard() here otherwise when we torch.load
    a diff sing itll try to init a tensorboard in our location before
    we've chdired into a better place
    """
    pass
  def set_tensorboard(self):
    print("intializing tensorboard")
    self.w = SummaryWriter(
        log_dir='tb',
        max_queue=1,
    )
    print("initialized writer for", self.name)
  def which(self, no_yaml=False):
    return which(self.cfg,no_yaml)
  def yaml(self):
    return yaml(self.cfg)
  def tb_scalar(self, plot_name, val, j=None):
    """
    Best way to write a scalar to tensorboard.
     * feed in `j=` to override it otherwise it'll use `sing.s.j`
     * include a '/' in the plot_name like 'Validation/MyModel' to override
      * the default behavior where it assumes you want 'Validation' to mean
      * 'Validation/{cls_name(sing.model)}'
    """

    if '/' not in plot_name:
      plot_name = f'{plot_name}/{cls_name(sing.model)}'

    if j is None:
      j = sing.train_state.j
    
    self.w.add_scalar(plot_name, val, j)
    self.w.flush()
  def tb_plot(self, plot_name, fig, j=None):
    """
    Best way to write a plot to tensorboard.
     * feed in `j=` to override it otherwise it'll use `sing.s.j`
     * include a '/' in the plot_name like 'Validation/MyModel' to override
      * the default behavior where it assumes you want 'Validation' to mean
      * 'Validation/{cls_name(sing.model)}'
    """

    if '/' not in plot_name:
      plot_name = f'{plot_name}/{cls_name(sing.model)}'

    if j is None:
      j = sing.train_state.j
    
    self.w.add_scalar(plot_name, val, j)
    self.w.flush()
    
    



class Stats:
  """
  This is what `sing.stats` is.
  Used for stuff like tracking amount of concrete evaluation that happens
  or whatever else a model might want. model.__init__ should modify `sing.stats`
  for example `sing.stats.concrete_count=0` then other places in the codebase
  can increment that value.
  """
  def print_stats(self):
    print("Stats:")
    for k,v in self.__dict__.items():
      if callable(v):
        v = v(self)
      print(f'\t{k}: {v}')

# note we must never overwrite this Sing. We should never do `matt.sing.sing = ...` 
# because then it would only overwrite the local version and the version imported by other modules
# would stay as the old verison. I tested this.
sing = Sing()