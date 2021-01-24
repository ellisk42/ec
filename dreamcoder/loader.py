from dreamcoder.domains.list.makeDeepcoderData import DeepcoderTaskloaderInner
from dreamcoder.grammar import Grammar
from dreamcoder.domains.misc.deepcoderPrimitives import deepcoderPrimitives,deepcoderPrimitivesPlusPlus
from dreamcoder.matt.sing import sing

class Taskloader:
  def __init__(self, train=False, test=False, valid=False) -> None:
    pass
  def grammar(self):
    # return a Grammar object
    raise NotImplementedError
  def valid(self):
    # return all validation tasks in a list
    raise NotImplementedError

class DeepcoderTaskloader(Taskloader):
  def __init__(self, train=False, test=False, valid=False) -> None:
    self.trainloader = self.valid_frontiers = self.testloader = None
    if train:
      self.trainloader = DeepcoderTaskloaderInner(mode='train')
    if valid:
      self.valid_frontiers = DeepcoderTaskloaderInner(mode='valid').getTasks(sing.cfg.loader.max_valid)
    if test:
      self.testloader = DeepcoderTaskloaderInner(mode='test')

    prims = deepcoderPrimitivesPlusPlus() if cfg.data.expressive_lambdas else deepcoderPrimitives()
    self.g_lambdas = self.inner.g_lambdas
    self.g = Grammar.uniform(prims, g_lambdas = taskloader.g_lambdas)
    sing.num_exs = sing.cfg.data.N
  def grammar(self):
    return self.g
  def get_valid(self):
    if self.valid_frontiers is None:
      raise ValueError("wasnt created to give out valid frontiers")
    return self.valid_frontiers

  