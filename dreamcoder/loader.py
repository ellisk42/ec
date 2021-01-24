from dreamcoder.domains.list.makeDeepcoderData import DeepcoderTaskloaderInner
from dreamcoder.grammar import Grammar
from dreamcoder.domains.misc.deepcoderPrimitives import deepcoderPrimitives,deepcoderPrimitivesPlusPlus
from dreamcoder.matt.sing import sing

class Taskloader:
  def __init__(self, train=False, test=False, valid=False) -> None:
    self.train = train
    self.test = test
    self.valid = valid
  def valid_tasks(self):
    """
    Return a list of validation tasks.
    Often you'll want this to always return the same list since
    we usually just want a small set of validation tasks and never
    really need to read them from the disk except during the Taskloader
    __init__.

    This doesnt take `n` bc it's assumed you want all `cfg.loader.max_valid` tasks

    Ideally these validation tasks should come from a separate file than
    test/train tasks but theres no hard restriction on that (eg if you
    dont actually guide training/selection of your model using the
    validation set then you could reasonably set it to just come
    from the same spot as the test tasks)
    """
    raise NotImplementedError
  def train_tasks(self,n=None,exact=False):
    """
    Return a list of training tasks.
    `n` puts a max on the number of tasks returned (may return less).
    if `n=None` then n is set to `cfg.loader.buf_size` 
    `n` can be larger than buf_size and in that case `n` tasks will be returned
    `exact=True` would mean it errors if fewer than n tasks would be returned
    """
    raise NotImplementedError
  def test_tasks(self,n=None,exact=False):
    """
    Return a list of testing tasks.
    `n` puts a max on the number of tasks returned (may return less).
    if `n=None` then n is set to `cfg.loader.buf_size` 
    `n` can be larger than buf_size and in that case `n` tasks will be returned
    `exact=True` would mean it errors if fewer than n tasks would be returned
    """
    raise NotImplementedError
  def _check_this_obj(self,exact=False):
    """
    Call this at the end of __init__ please
    """
    assert hasattr(self,'g'), "Please assign self.g in __init__"


class DeepcoderTaskloader(Taskloader):
  def __init__(self, train=False, test=False, valid=False) -> None:
    super().__init__(train,test,valid)

    self.trainloader = self.valid_frontiers = self.testloader = None

    if train:
      self.trainloader = DeepcoderTaskloaderInner(mode='train')
    if valid:
      self._valid_tasks = DeepcoderTaskloaderInner(mode='test').getTasks(sing.cfg.loader.max_valid)
    if test:
      self.testloader = DeepcoderTaskloaderInner(mode='test')

    prims = deepcoderPrimitivesPlusPlus() if sing.cfg.data.expressive_lambdas else deepcoderPrimitives()
    self.g_lambdas = self.inner.g_lambdas
    self.g = Grammar.uniform(prims, g_lambdas = taskloader.g_lambdas)
    sing.num_exs = sing.cfg.data.N

    self._check_this_obj()

  def valid_tasks(self):
    """
    Return a list of validation tasks.
    Often you'll want this to always return the same list since
    we usually just want a small set of validation tasks and never
    really need to read them from the disk except during the Taskloader
    __init__.

    This doesnt take `n` bc it's assumed you want all `cfg.loader.max_valid` tasks

    Ideally these validation tasks should come from a separate file than
    test/train tasks but theres no hard restriction on that (eg if you
    dont actually guide training/selection of your model using the
    validation set then you could reasonably set it to just come
    from the same spot as the test tasks)
    """
    return self._valid_tasks

  def train_tasks(self,n=None,exact=False):
    """
    Return a list of training tasks.
    `n` puts a max on the number of tasks returned (may return less).
    if `n=None` then n is set to `cfg.loader.buf_size` 
    """
    res = self.trainloader.getTasks(n)
    if exact and len(res) != n:
      raise ValueError
    return res

  def test_tasks(self,n=None,exact=False):
    """
    Return a list of testing tasks.
    `n` puts a max on the number of tasks returned (may return less).
    if `n=None` then n is set to `cfg.loader.buf_size` 
    """
    res = self.testloader.getTasks(n)
    if exact and len(res) != n:
      raise ValueError
    return res

  