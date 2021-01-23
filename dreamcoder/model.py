
import torch
from torch import nn
from dreamcoder.matt.sing import sing

class MBAS(nn.Module):
  def __init__(self,vhead,phead,em,g):
    super().__init__()
    self.vhead = vhead
    self.phead = phead
    self.em = em
    self.g = g

    self.running_vloss = RunningLoss()
    self.running_ploss = RunningLoss()

    self.optim = torch.optim.Adam(self.parameters(), lr=sing.cfg.optim.lr, eps=1e-3, amsgrad=True)
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

    print(f'\tValueLoss {sing.vhead.cls_name} {self.running_vloss.avg()}')
    print(f'\tPolicyLoss {sing.phead.cls_name} {self.running_ploss.avg()}')

    self.running_ploss.reset()
    self.running_vloss.reset()
  def valid_step(self,fs):
    self.eval()
    with torch.no_grad():
      running_loss = RunningValue()
      running_vloss = RunningValue()
      running_ploss = RunningValue()
      for f in fs:
        vloss = self.vhead.valueLossFromFrontier(f, self.g)
        ploss = self.phead.policyLossFromFrontier(f, self.g)
        running_vloss.add(vloss)
        running_ploss.add(ploss)
        running_loss.add(vloss+ploss)
    
    sing.tb_scalar('ValidPolicyLoss', self.running_ploss.avg())
    sing.tb_scalar('ValidValueLoss', self.running_vloss.avg())

    to_print = f'\tValidValueLoss {sing.vhead.cls_name} {self.running_vloss.avg()}\n\tValidPolicyLoss {sing.phead.cls_name} {self.running_ploss.avg()}'

    return running_loss.avg(), to_print


