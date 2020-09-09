

#train robustfill baseline


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

import string
from dreamcoder.Astar import Astar
from likelihoodModel import AllOrNothingLikelihoodModel
from dreamcoder.policyHead import RNNPolicyHead, BasePolicyHead, REPLPolicyHead
from dreamcoder.domains.tower.makeTowerTasks import makeNewMaxTasks
from dreamcoder.SMC import SMC

from dreamcoder.valueHead import RBPrefixValueHead
#"rbPolicyOnlyBigram_SRE=True.pickle"
from dreamcoder.frontier import Frontier, FrontierEntry

from syntax_robustfill import SyntaxCheckingRobustFill


prims = robustFillPrimitives()
g = Grammar.uniform(prims)

def stringify(line):
    lst = []
    string = ""
    for char in line+" ":
        if char == " ":
            if string != "":
                lst.append(string)
            string = ""
        elif char in '()':
            if string != "":
                lst.append(string)
            string = ""
            lst.append(char)
        else:
            string += char      
    return lst

def getDatum(n_ex):
    #tsk = random.choice(tasks)
    #tp = tsk.request
    p, task = r.recognitionModel.featureExtractor.sampleHelmholtzTask(arrow(texpression, texpression))
    #p = g.sample(tp, maximumDepth=6)
    #task = fe.taskOfProgram(p, tp)

    del task.examples[n_ex:]
    #print(len(task.examples))
    ex = makeExamples(task)
    return ex, stringify(str(p))

def makeExamples(task):    
    examples = []
    for x, y in task.examples:
        examples.append((list(x),list(y)))
    #print(examples)
    return examples

parser = argparse.ArgumentParser()
parser.add_argument('--num_training_programs', type=int, default=20000000, help='number of episodes for training')
parser.add_argument('--batchsize', type=int, default=32 )
parser.add_argument('--use_saved_val', action='store_true', help='use saved validation problems')
parser.add_argument('--save_path', type=str, default='robustfill_baseline0.p')
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=50)
parser.add_argument('--save_old_freq', type=int, default=10000)
parser.add_argument('--positional', action='store_true')
parser.add_argument('--gpu', type=int, default=None)
args = parser.parse_args()

if args.gpu is not None:
    torch.cuda.set_device(args.gpu)
batchsize = args.batchsize


#sys.setrecursionlimit(50000)
graph = ""
ID = 'rb'
runType = "PolicyOnly" #"Policy"
#runType =""
model = "Bigram"
useREPLnet = False
path = f'experimentOutputs/{ID}{runType}{model}_SRE=True{graph}.pickle'

print(path)
with open(path, 'rb') as h:
    r = dill.load(h)

fe = r.recognitionModel.featureExtractor

extras = ['(', ')', 'lambda'] + ['$'+str(i) for i in range(10)]


input_vocabularies = [list(string.printable[:-4]) + ['EOE'], string.printable[:-4]]
target_vocabulary = [str(p) for p in g.primitives] + extras

m = SyntaxCheckingRobustFill(input_vocabularies=input_vocabularies,
                            target_vocabulary=target_vocabulary)
m.cuda()
m.iter = 0


t = time.time()
for i in range(int(args.num_training_programs/args.batchsize)):

    batch = [getDatum(4) for _ in range(batchsize)]
    inputs, targets = zip(*batch)

    score, syntax_score = m.optimiser_step(inputs,targets) #syntax or not, idk
    m.iter += 1

    print(f"total time: {time.time() - t}, total num ex processed: {(i+1)*batchsize}, avg time per ex: {(time.time() - t)/((i+1)*batchsize)}, score: {score}")

    if i%args.save_freq==0:
        torch.save(m, args.save_path)
        print('saved model')
    if i%args.save_old_freq==0:
        torch.save(m, args.save_path+str(m.iter))