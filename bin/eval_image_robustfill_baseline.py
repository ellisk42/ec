#eval_robustfill_baseline.py

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

from train_image_robustfill_baseline import makeExamples
from dreamcoder.SMC import SearchResult
from dreamcoder.utilities import ParseFailure

def check_candidate(task, raw_candidate):
    p = None
    #print(raw_candidate)
    try:
        p = Program.parse(" ".join(raw_candidate))
        #print(p)
        ll = task.logLikelihood(p, timeout=1)
        
        if ll == 0.0:
            return True, p
        return False, p
    except IndexError: return False, p
    except AssertionError: return False, p
    except ParseFailure: return False, p
    except TypeError: return False, p
    except KeyError: return False, p

def get_num_nodes(raw_candidate):
    return len( [x for x in raw_candidate if x not in ['(', ')', 'lambda']] )

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=16 )
parser.add_argument('--path', type=str, default='image_robustfill_baseline0.p')
parser.add_argument('--results_path', type=str, default='image_robustfill_baseline_results.p')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--timeout', type=int, default=30)
parser.add_argument('--tasks', type=str, default='kevin')
args = parser.parse_args()

if args.gpu is not None:
    torch.cuda.set_device(args.gpu)
batchsize = args.batchsize


m = torch.load(args.path)
m.cuda()
m.eval()
m.max_length = 50


graph = ""
ID = 'towers' + str(3)
runType = "PolicyOnly" #"Policy"
#runType =""
model = "RNN" #"Sample"
#useREPLnet = args.useREPLnet
#useRLValue = False
path = f'experimentOutputs/{ID}{runType}{model}_SRE=True{graph}.pickle'
print(path)
with open(path, 'rb') as h:
    r = dill.load(h)


#assert 0
times = []
# ttasks = r.getTestingTasks()

# if args.tasks == 'kevin':
#     ttasks = makeOldTasks(synth=True, challenge=True)
# elif args.tasks == 'synth':
#     ttasks = makeOldTasks(synth=True, challenge=False)
# elif args.tasks == 'challenge':
#     ttasks = makeOldTasks(synth=False, challenge=True)
# #ttasks = makeNewMaxTasks() + r.getTestingTasks()

ttasks = makeNewMaxTasks() + r.getTestingTasks()
print("number of tasks", len(ttasks))


#ttasks, lls = makeRandomTasks(10, r)
print("number of tasks", len(ttasks))
ttasks = list(set(ttasks))
print("number of tasks", len(ttasks))

#print("using max tasks and old tasks")


#import pdb; pdb.set_trace()
nhit = 0
stats = {}
nums = {}
for i, t in enumerate(ttasks):
    with torch.no_grad():
        print("****NEW TASK****")
        tasks = [t]
        #tasks = []
        print(tasks)
        #print("ll is", lls[i])

        #likelihoodModel = AllOrNothingLikelihoodModel(timeout=0.01)
        #tasks = [frontier.task]

        # g = r.recognitionModel.grammarOfTask(tasks[0]).untorch()
        # fs, searchTimes, totalNumberOfPrograms, reportedSolutions = solver.infer(g, tasks, likelihoodModel, 
        #                                     timeout=30,
        #                                     elapsedTime=0,
        #                                     evaluationTimeout=0.01,
        #                                     maximumFrontiers={tasks[0]: 2},
        #                                     CPUs=1,
        #                                     ) 
        inputs = makeExamples(t)
        start = time.time()
        totalNumberOfPrograms = 0
        reportedSolutions = {t:[]}
        searchTimes = {t:None}
        hit = False
        while time.time() - start < args.timeout and not hit:
            candidates, scores = m.sample([inputs]*args.batchsize, returnScore=True)
            for candidate, score in zip(candidates, scores):
                #print(candidate)
                totalNumberOfPrograms += get_num_nodes(candidate)
                hit, p = check_candidate(t, candidate)
                if hit:
                    dt =  time.time() - start
                    searchTimes = {t:dt}                
                    reportedSolutions[t].append(SearchResult(p, score, dt, totalNumberOfPrograms))
                    break

        print("done")
        print("total prog", totalNumberOfPrograms)  
        print("searchTimes", searchTimes, flush=True)
        print()
        print()
        
        if list(searchTimes.values())[0]:
            nhit += 1

        for k, v in reportedSolutions.items():
            stats[k] = v

        nums[tasks[0]] = totalNumberOfPrograms

print("n hit:", nhit)

class PseudoResult:
    pass 
pseudoResult = PseudoResult()
pseudoResult.testingSearchStats = [stats]
pseudoResult.testingNumOfProg = [nums]

savePath = args.results_path
with open(savePath, 'wb') as h:
    dill.dump(pseudoResult, h)
print("saved at", savePath)