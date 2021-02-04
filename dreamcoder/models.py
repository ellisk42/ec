from dreamcoder.matt.util import *
import torch
from torch import nn
from dreamcoder.matt.sing import sing
from dreamcoder.SMC import SMC
from dreamcoder.Astar import Astar
from dreamcoder import valueHead,policyHead
from dreamcoder.matt import plot
from dreamcoder import aux_models
from dreamcoder.likelihoodModel import AllOrNothingLikelihoodModel

class MBAS(nn.Module):
  def __init__(self):
    super().__init__()

    sing.stats.call_encode_known_ctx = 0
    sing.stats.call_encode_exwise = 0
    sing.stats.fn_called_concretely = 0
    sing.stats.fn_called_abstractly = 0
    self.running_vloss = RunningFloat()
    self.running_ploss = RunningFloat()


    # add all submodules required by vhead and phead
    submodules = set()

    submodules |= {
      'repl': {aux_models.AbstractionFn, aux_models.AbstractTransformers, aux_models.AbstractComparer},
      'rnn': {aux_models.AbstractionFn, aux_models.ProgramRNN},
      'check_invalid': set(),
      'uniform': set(),
    }[sing.cfg.model.vhead]

    submodules |= {
      'repl': {aux_models.AbstractionFn, aux_models.AbstractTransformers, aux_models.AbstractComparer},
      'rnn': {aux_models.AbstractionFn, aux_models.ProgramRNN},
      'uniform': set(),
    }[sing.cfg.model.phead]

    for mod in submodules:
      # initialize submodules and assign as attributes to self
      name = {
        aux_models.AbstractionFn: 'abstraction_fn',
        aux_models.AbstractTransformers: 'abstract_transformers',
        aux_models.AbstractComparer: 'abstract_comparer',
        aux_models.ProgramRNN: 'program_rnn',
      }[mod]
      setattr(self,name,mod())

    # vhead and phead init

    self.vhead = {
      'repl': valueHead.ListREPLValueHead,
      'rnn': valueHead.RNNValueHead,
      'check_invalid': valueHead.InvalidIntermediatesValueHead,
      'uniform': valueHead.UniformValueHead,
    }[sing.cfg.model.vhead]()

    self.phead = {
      'repl': policyHead.ListREPLPolicyHead,
      'rnn': policyHead.RNNPolicyHead,
      'uniform': policyHead.UniformPolicyHead,
    }[sing.cfg.model.phead]()

    # init the optimizer
    self.optim = torch.optim.Adam(self.parameters(), lr=sing.cfg.optim.lr, eps=1e-3, amsgrad=True)

  def run_tests(self,fs):
    # fs are the validation frontiers for the record
    self.abstraction_fn.encoder.run_tests()
    # TODO run more tests here!

  def train_step(self,fs):
    assert len(fs) == 1
    [f] = fs
    self.train()
    self.zero_grad()

    vloss = self.vhead.train_loss(f.p, f.t)
    ploss = self.phead.train_loss(f.p, f.t)

    self.running_vloss.add(vloss)
    self.running_ploss.add(ploss)

    loss = vloss + ploss
    loss.backward()
    self.optim.step()
    return loss, None

  def print_every(self):
    sing.tb_scalar('PolicyLoss', self.running_ploss.avg())
    sing.tb_scalar('ValueLoss', self.running_ploss.avg())

    print(f'\tValueLoss {cls_name(self.vhead)} {self.running_vloss.avg()}')
    print(f'\tPolicyLoss {cls_name(self.phead)} {self.running_ploss.avg()}')

    self.running_ploss.reset()
    self.running_vloss.reset()

    sing.stats.print_stats()

  def valid_step(self,fs):
    self.eval()
    with torch.no_grad():
      running_loss = RunningFloat()
      running_vloss = RunningFloat()
      running_ploss = RunningFloat()
      for f in fs:
        vloss = self.vhead.train_loss(f.p,f.t)
        ploss = self.phead.train_loss(f.p,f.t)
        running_vloss.add(vloss)
        running_ploss.add(ploss)
        running_loss.add(vloss+ploss)
    
    sing.tb_scalar('ValidPolicyLoss', self.running_ploss.avg())
    sing.tb_scalar('ValidValueLoss', self.running_vloss.avg())

    to_print = f'\tValidValueLoss {cls_name(self.vhead)} {self.running_vloss.avg()}\n\tValidPolicyLoss {cls_name(self.phead)} {self.running_ploss.avg()}'

    return running_loss.avg(), to_print

  def search(self, fs, timeout, verbose=True):
    """
    Gets `solver` from sing.cfg.solver
    Note that AStar and SMC will also read their details from `sing.cfg.test` for
      stuff like no_resample
    """
    assert len(fs) > 0

    if sing.cfg.test.scaffold:
      raise NotImplementedError # see the old test.py:test_models() if you want to impl this

    likelihood_model = AllOrNothingLikelihoodModel(timeout=0.01)
    solver = {
      'astar': Astar,
      'smc': SMC,
    }[sing.cfg.solver.type](self.phead, self.vhead, sing.cfg.solver)

    starting_nodes = None # related to cfg.test.scaffold

    search_tries = []

    self.eval()
    with torch.no_grad():
      for i,f in enumerate(fs):
        t = f.t
        _fs, times, num_progs, solns = solver.infer(
          sing.g,
          [t],
          likelihood_model,
          timeout=timeout,
          starting_nodes=starting_nodes
        )
        solns = solns[t]
        time = times[t]
        if len(solns) == 0:
          search_tries.append(plot.SearchTry(timeout,num_progs,None))
          if verbose:
            red(f"[{i+1}/{len(fs)}] failed to solve {t.name} (searched {num_progs} programs)")
        else:
          assert len(solns) == 1
          [soln] = solns
          search_tries.append(plot.SearchTry(time,num_progs,soln))
          if verbose:
              green(f"[{i+1}/{len(fs)}] solved {t.name} in {time:.2f}s (searched {num_progs} programs)")
              # t,d,s = get_depth(solns[0].program)
              # print(f"\t-> [T{t}d{d}s{s}] {soln.program}")
              print(f"\t-> {soln.program}")
    
    if verbose:
      blue(f'solved {len(search_tries)}/{len(fs)} tasks ({len(search_tries)/len(fs)*100:.1f}%)\n')
    model_result = plot.ModelResult(search_tries,timeout)
    return model_result
    
class Deepcoder(nn.Module):
  pass # todo
class Robustfill(nn.Module):
  pass # todo