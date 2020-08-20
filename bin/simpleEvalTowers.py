#simpleEvalTowers.py
import dill
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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--useValue', action='store_true')
parser.add_argument('--useREPLnet', action='store_true')
args = parser.parse_args()


def test_policyTiming():
    from dreamcoder.Astar import Astar
    from likelihoodModel import AllOrNothingLikelihoodModel
    from dreamcoder.policyHead import RNNPolicyHead, BasePolicyHead, REPLPolicyHead
    from dreamcoder.domains.tower.makeTowerTasks import makeNewMaxTasks
    from dreamcoder.valueHead import SampleDummyValueHead

    sys.setrecursionlimit(50000)
    graph = ""
    ID = 'towers' + str(3)
    runType = "" #"PolicyOnly" #"Policy"
    #runType =""
    model = "RNN" #"Sample"
    useREPLnet = args.useREPLnet
    useValue = True

    torch.cuda.set_device(0)


    path = f'experimentOutputs/{ID}{runType}{model}_SRE=True{graph}.pickle'
    print(path)
    with open(path, 'rb') as h:
        r = dill.load(h)
    
    # useREPLnet = True
    # model = "REPL"

    if not hasattr(r.recognitionModel, "policyHead"):
        r.recognitionModel.policyHead = BasePolicyHead()

    if useREPLnet:
        path=f"experimentOutputs/{ID}PolicyOnlyREPL.pickle_RecModelOnly"
        with open(path, 'rb') as h:
            repl = torch.load(h)
        model = "REPL"
        #import pdb; pdb.set_trace()

        r.recognitionModel = repl
        #print(r.recognitionModel.policyHead)

    if useValue:
        path =f"experimentOutputs/{ID}PolicyOnly{model}.pickle_RecModelOnlyrl"
        with open(path, 'rb') as h:
            recModel = torch.load(h)
        print("using rec model:", path)
        r.recognitionModel = recModel
        r.recognitionModel.cuda()
        if not args.useValue:
            r.recognitionModel.valueHead = SampleDummyValueHead()


    g = r.grammars[-1]
    print(r.recognitionModel.gradientStepsTaken)
    solver = r.recognitionModel.solver

    solver = Astar(r.recognitionModel)

    times = []
    ttasks = r.getTestingTasks()


    ttasks = makeNewMaxTasks() + r.getTestingTasks()

    #import pdb; pdb.set_trace()
    print("using max tasks")

    #r.recognitionModel.policyHead.cpu()
    ttasts = list(set(ttasks))
    print("number of tasks:", len(ttasks))

    nhit = 0
    stats = {}
    nums = {}
    for i, t in enumerate(ttasks):
        #if i > 20: break
        tasks = [t]
        print(tasks)
        likelihoodModel = AllOrNothingLikelihoodModel(timeout=0.01)
        #tasks = [frontier.task]

        g = r.recognitionModel.grammarOfTask(tasks[0]).untorch()
        fs, searchTimes, totalNumberOfPrograms, reportedSolutions = solver.infer(g, tasks, likelihoodModel, 
                                            timeout=30,
                                            elapsedTime=0,
                                            evaluationTimeout=0.01,
                                            maximumFrontiers={tasks[0]: 2},
                                            CPUs=1,
                                            ) 
        print("done")
        print("total prog", totalNumberOfPrograms)  
        print("searchTimes", searchTimes)
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

    savePath = f'experimentOutputs/{ID}{runType}PseudoResult{model}RLValue={useValue}_SRE=True.pickleDebug'
    with open(savePath, 'wb') as h:
        dill.dump(pseudoResult, h)
    print("saved at", savePath)

if __name__=='__main__':
    test_policyTiming()