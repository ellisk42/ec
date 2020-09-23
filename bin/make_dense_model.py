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


path = 'experimentOutputs/rbDenseREPLNoConcrete.pickle_RecModelOnlydict'

d = torch.load(path)


from dreamcoder.policyHead import RBREPLPolicyHead
from dreamcoder.domains.rb.main import RBFeatureExtractor
from collections import OrderedDict

fe = RBFeatureExtractor(cuda=True, char_embed_dim=20, useConvs=False)
pdict = OrderedDict( (key[11:], value) for key, value in d.items() if 'policyHead' in key)

g = Grammar.uniform(robustFillPrimitives())                                                                                                                                      
policyHead = RBREPLPolicyHead(g, fe, fe.H,  canonicalOrdering=True, noConcrete=True, useConvs=False)                                                                             
policyHead.load_state_dict(pdict)



torch.save(policyHead, 'experimentOutputs/rbDenseREPLNoConcrete.pickle_RecModelOnlyPolicyHead')



"""
steps:
make state_dict on om:
run bin/make_dense_state_dict.py on ome
scp 'experimentOutputs/rbDenseREPLNoConcrete.pickle_RecModelOnlydict' to moe
run bin/make_dense_model.py

python bin/simpleEval.py --tasks challenge --model "REPLNoConcrete" --runType Dense --useHead 'experimentOutputs/rbDenseREPLNoConcrete.pickle_Rec
ModelOnlyPolicyHead'



"""