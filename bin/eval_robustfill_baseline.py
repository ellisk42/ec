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

from train_robustfill_baseline import makeExamples
from dreamocoder.SMC import SearchResult
from dreamcoder.utilities import ParseFailure

def check_candidate(task, raw_candidate):
    try:
        p = Program.parse(" ".join(raw_candidate))
        #print(p)
    except ParseFailure: return False
    except IndexError: return False
    except AssertionError: return False

    ll = task.logLikelihood(p, timeout=1)
    if ll == 0.0:
    	return True
    return False

def get_num_nodes(raw_candidate):
	return len( [x for x in raw_candidate if x not in ['(', ')', 'lambda']] )

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=32 )
parser.add_argument('--path', type=str, default='robustfill_baseline0.p')
parser.add_argument('--path', type=str, default='robustfill_baseline0.p')
parser.add_argument('--results_path', type=str, default='robustfill_baseline_results.p')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--timeout', type=int, default=120)
args = parser.parse_args()

if args.gpu is not None:
    torch.cuda.set_device(args.gpu)
batchsize = args.batchsize


m = torch.load(args.path)
m.cuda()
m.max_length = 50


graph = ""
ID = 'rb'
runType = "PolicyOnly" #"Policy"
#runType =""
model = "Bigram"
problemPath = f'experimentOutputs/{ID}{runType}{model}_SRE=True{graph}.pickle'

print("problems from:", problemPath)
with open(path, 'rb') as h:
    r = dill.load(h)


times = []
ttasks = r.getTestingTasks()
ttasks = makeOldTasks(synth=False, challenge=True)
#ttasks = makeNewMaxTasks() + r.getTestingTasks()

print("number of tasks", len(ttasks))

from bin.searchGraphs import FILTER_OUT
tnames = []
for ins, outs in FILTER_OUT:
    examples = list(zip(ins, outs))
    name = str(examples[0])
    tnames.append(name)

ttasks = [t for t in ttasks if t.name not in tnames]

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
    while time.time() - start < args.timeout and not hit:
    	candidates, scores, _ = m.sample([inputs]*args.batchsize)
    	for candidate, score in zip(candidates, scores):
    		totalNumberOfPrograms += get_num_nodes(candidate)
    		hit = check_candidate(t, candidate)
    		if hit:
    			dt =  time.time() - start
    			searchTimes = {t:dt}    			
    			reportedSolutions[t].append(SearchResult(p, score, dt, totalNumberOfPrograms))
    			break

    print("done")
    print("total prog", totalNumberOfPrograms)  
    print("searchTimes", searchTimes)
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