#make dense model...
#simpleEval.py
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.grammar import *
from dreamcoder.domains.arithmetic.arithmeticPrimitives import *
from dreamcoder.domains.list.listPrimitives import *
from dreamcoder.program import Program
from dreamcoder.valueHead import *
from dreamcoder.zipper import *

from dreamcoder.domains.tower.towerPrimitives import *
import time
import torch
import dill

from dreamcoder.domains.rb.rbPrimitives import *

from dreamcoder.domains.rb.main import makeOldTasks, makeTasks
import argparse


path = 'experimentOutputs/rbDenseREPLNoConcrete.pickle_RecModelOnly'

d = torch.load(path)


path_dict = path + 'dict'
torch.save(m.state_dict(), path_dict)  