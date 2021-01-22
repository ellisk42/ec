
"""
`sing` as in the singleton design pattern. See this page on "How do I share global variables across modules?" from the python docs:
  https://docs.python.org/3/faq/programming.html#id11

Nothing in this file is initialized because we want it to be set manually by whatever loading or init code gets run.
"""
import torch.nn as nn
import functools

class Sing:
  def __init__(self) -> None:
    self.cfg = None
    self.em = None
    self.vhead = None
    self.phead = None
    self.heads = None
    self.num_exs = None
    self.to_optimize = None
    self.track = StatTrack()
  def from_sing(self,s):
      for k,v in vars(s).items():
          setattr(self,k,v)


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

sing = Sing() # this is the global Sing instance. Import with `from dreamcoder.matt.sing import sing`
