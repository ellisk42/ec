
"""
`sing` as in the singleton design pattern. See this page on "How do I share global variables across modules?" from the python docs:
  https://docs.python.org/3/faq/programming.html#id11

Nothing in this file is initialized because we want it to be set manually by whatever loading or init code gets run.
"""
from dreamcoder.matt.util import *
import torch.nn as nn
import torch
import functools
from torch.utils.tensorboard import SummaryWriter
from dreamcoder import model,loader

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
          raise ValueError("can't do mode=test without a file to load from")

        self.cfg = cfg
        from dreamcoder.matt.train import TrainState
        self.train_state = TrainState(cfg)
        self.cwd = getcwd()
        self.name = f'{cfg.job_name}.{cfg.run_name}'
        self.device = torch.device(cfg.device)

        self.taskloader = {
          'deepcoder':loader.DeepcoderTaskloader
        }[self.cfg.data.type](train=True,valid=True)

        self.g = self.taskloader.g

        self.model = {
          'mbas': model.MBAS,
          'dc': model.Deepcoder, # todo
          'rb': model.Robustfill, # todo
        }[cfg.model.type]()

        self.model.to(self.device)
        self.set_tensorboard(self.name)
      else:
        ############################################
        ### * LOAD OLD SING CONTAINING A MODEL * ###
        ############################################
        overrided = [arg.split('=')[0] for arg in sys.argv[1:]]
        device = torch.device(cfg.device) if 'device' in overrided else None
        # in these cases cfg.load points to a Sing file

        path = sing_path_from_regex(cfg.load)
        _sing = torch.load(path,map_location=device) # `None` means stay on original device
        self.clone_from(_sing) # will even copy over stuff like SummaryWriter object so no need for post_load()
        del _sing
        if device is not None:
          self.device = device # override sings device indicator used in various places
        self.apply_overrides(overrided,cfg)
        print(f"chdir to {self.cwd}")
        os.chdir(self.cwd)
        self.set_tensorboard(self.name)
    elif cfg.mode in ('plot','testgen','cmd'):
      ######################################################################
      ### * BARE BONES SING: NO TRAIN_STATE, NO MODEL, NO LOADERS, ETC * ###
      ######################################################################
      self.cfg = cfg
    else:
      raise ValueError(f"mode={cfg.mode} is not a valid mode")

   
  def save(self, name):
      path = saves_path() / f'{name}.sing'
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
  def set_tensorboard(self,log_dir):
    print("intializing tensorboard")
    self.w = SummaryWriter(
        log_dir=self.name,
        max_queue=1,
    )
    print("initialized writer for", self.name)
  def which(self):
    return which(self.cfg)
  def yaml(self):
    return yaml(self.cfg)


class StatTrack():
    def __init__(self) -> None:
        super().__init__()
        self.total_ct = 0
        self.abstract_ct = 0
    def concrete_ratio(self):
        if self.total_ct == 0:
            return -1 # untracked or no data yet
        return 1-(self.abstract_ct / self.total_ct)
    def track_concrete_ratio(self,propagate):
        @functools.wraps(propagate)
        def _propagate(node,*args,**kwargs):
            res = propagate(node,*args,**kwargs)
            if not node.isOutput and not node.isAbstraction: # bc these dont count towards size() either
              if res._abstract is not None:
                self.abstract_ct += 1
            if node.isOutput:
              self.total_ct += node.size()
            return res
        return _propagate

class ToOptimize(nn.Module):
  def __init__(self,modules:list):
    super().__init__()
    self.modules = nn.ModuleList(modules)

# note we must never overwrite this Sing. We should never do `matt.sing.sing = ...` 
# because then it would only overwrite the local version and the version imported by other modules
# would stay as the old verison. I tested this.
sing = Sing()