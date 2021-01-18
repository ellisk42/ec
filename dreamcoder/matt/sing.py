
"""
`sing` as in the singleton design pattern. See this page on "How do I share global variables across modules?" from the python docs:
  https://docs.python.org/3/faq/programming.html#id11

Nothing in this file is initialized because we want it to be set manually by whatever loading or init code gets run.
"""
cfg = None
em = None
vhead = None
phead = None
heads = None
num_exs = None

import functools
class StatTrack():
    def __init__(self) -> None:
        super().__init__()
        self.concrete_ct = 0
        self.abstract_ct = 0
    def concrete_ratio(self):
        if self.concrete_ct + self.abstract_ct == 0:
            return None # untracked or no data yet
        return self.concrete_ct / (self.concrete_ct + self.abstract_ct)
    def track_concrete_ratio(self,propagate):
        @functools.wraps(propagate)
        def _propagate(*args,**kwargs):
            res = propagate(*args,**kwargs)
            if res._abstract is None:
                self.concrete_ct += 1
            else:
                self.abstract_ct += 1
            return res
        return _propagate

track = StatTrack()