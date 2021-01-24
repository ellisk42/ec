from dreamcoder.matt.util import *
import torch
from torch import nn
from dreamcoder.matt.sing import sing
from dreamcoder.SMC import SMC
from dreamcoder.Astar import Astar
from dreamcoder.em import ExecutionModel
from dreamcoder import valueHead,policyHead

class MBAS(nn.Module):
  def __init__(self):
    super().__init__()

    sing.stats.call_encode_known_ctx = 0
    sing.stats.call_encode_exwise = 0
    sing.stats.fn_called_concretely = 0
    sing.stats.fn_called_abstractly = 0

    self.em = ExecutionModel()

    self.vhead = {
      'repl': valueHead.ListREPLValueHead,
      'rnn': valueHead.SimpleRNNValueHead,
      'check_invalid': valueHead.InvalidIntermediatesValueHead,
      'uniform': valueHead.UniformValueHead,
    }[sing.cfg.model.vhead]()

    self.phead = {
      'repl': policyHead.ListREPLPolicyHead,
      'rnn': policyHead.RNNPolicyHead,
      'uniform': policyHead.UniformPolicyHead,
    }[sing.cfg.model.phead]()

    self.running_vloss = RunningFloat()
    self.running_ploss = RunningFloat()

    self.optim = torch.optim.Adam(self.parameters(), lr=sing.cfg.optim.lr, eps=1e-3, amsgrad=True)

  def run_tests(self,fs):
    # fs are the validation frontiers for the record
    self.em.extractor.run_tests()
    raise NotImplementedError # run some more tests!

  def train_step(self,fs):
    assert len(fs) == 1
    [f] = fs
    self.train()
    self.zero_grad()

    vloss = self.vhead.valueLossFromFrontier(f, self.g)
    ploss = self.phead.policyLossFromFrontier(f, self.g)

    self.running_vloss.add(vloss)
    self.running_ploss.add(ploss)

    loss = vloss + ploss
    loss.backward()
    self.optim.step()
    to_print = f'\tconcrete: {sing.track.concrete_ratio():.3f})'
    return loss, to_print

  def print_every(self):
    sing.tb_scalar('PolicyLoss', self.running_ploss.avg())
    sing.tb_scalar('ValueLoss', self.running_ploss.avg())

    print(f'\tValueLoss {cls_name(sing.vhead)} {self.running_vloss.avg()}')
    print(f'\tPolicyLoss {cls_name(sing.phead)} {self.running_ploss.avg()}')

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
        vloss = self.vhead.valueLossFromFrontier(f, self.g)
        ploss = self.phead.policyLossFromFrontier(f, self.g)
        running_vloss.add(vloss)
        running_ploss.add(ploss)
        running_loss.add(vloss+ploss)
    
    sing.tb_scalar('ValidPolicyLoss', self.running_ploss.avg())
    sing.tb_scalar('ValidValueLoss', self.running_vloss.avg())

    to_print = f'\tValidValueLoss {cls_name(sing.vhead)} {self.running_vloss.avg()}\n\tValidPolicyLoss {cls_name(sing.phead)} {self.running_ploss.avg()}'

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
    }[self.cfg.solver.type](self.phead, self.vhead, sing.cfg.solver)

    starting_nodes = None # related to cfg.test.scaffold

    search_tries = []

    self.eval()
    with torch.no_grad():
      for i,f in enumerate(fs):
        fs, times, num_progs, solns = solver.infer(
          self.g,
          [f],
          likelihood_model,
          timeout=timeout,
          starting_nodes=starting_nodes
        )
        solns = solns[f]
        times = times[f]
        if len(solns) == 0:
          search_tries.append(SearchTry(timeout,num_progs,None))
          if verbose:
            red(f"[{i+1}/{len(fs)}] failed to solve {f.name} (searched {num_progs} programs)")
        else:
          assert len(solns) == len(times) == 1
          [soln] = solns
          [time] = times
          search_tries.append(SearchTry(time,num_progs,soln))
          if verbose:
              green(f"[{i+1}/{len(fs)}] solved {f.name} in {time:.2f}s (searched {num_progs} programs)")
              raise NotImplementedError # reimplement get_depth t,d,s
              t,d,s = get_depth(solns[0].program)
              print(f"\t-> [T{t}d{d}s{s}] {soln.program}")
    
    if verbose:
      raise 
      blue(f'solved {len(search_tries)}/{len(fs)} tasks ({len(search_results)/len(test_tasks)*100:.1f}%)\n')
    model_result = plot.ModelResult(prefix=prefix,name=name, cfg=astar.owner.policyHead.cfg, search_results=search_results, search_failures=search_failures, timeout=timeout)

    