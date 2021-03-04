

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
from image_robustfill import Image_RobustFill


# prims = robustFillPrimitives()
# g = Grammar.uniform(prims)

def getInventionSpan(line, i):
    assert line[i] == '#'
    assert line[i+1] == '('

    inv = "#("
    count = 1
    for j in range(i+2, len(line)):
        if count == 0:
            return j - i, line[i:j] #Check

        if line[j] == "(": count+=1
        if line[j] == ")": count-=1

    assert False

def stringify(line):
    lst = []
    string = ""
    #for i in range(len(line) + 1) #char in line+" ":
    i = 0
    while i <= len(line):
        char = (line+" ")[i]
        if char == " ":
            if string != "":
                lst.append(string)
            string = ""
        elif char in '()':
            if string != "":
                lst.append(string)
            string = ""
            lst.append(char)
        elif char == "#": #if invention:
            l, inv = getInventionSpan(line, i)
            #print("inv", repr(inv))
            #print("l", l)
            i += l - 1
            string += inv

        else:
            string += char      
        i+=1
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


def getDatum():
    while True:
        if random.random() > args.hr:
            tsk = random.choice(tasks)
            tp = tsk.request
            p = g.sample(tp, maximumDepth=6)
            task = fe.taskOfProgram(p, tp)
        else:
            f = random.choice(nonEmptyFrontiers)
            task = f.task

            try:
                p = f.sample().program
            except:
                import pdb; pdb.set_trace()

        if task is None:
            #print("no taskkkk")
            continue
        ex = makeExamples(task)
        if ex is None: continue
        #print(p)
        #print(stringify(str(p)))
        #import pdb; pdb.set_trace()
        return ex, stringify(str(p))


def makeExamples(task):    
    return task.getImage().transpose(2,0,1)
    #return examples

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_training_programs', type=int, default=480000, help='number of episodes for training')
    parser.add_argument('--batchsize', type=int, default=32 )
    parser.add_argument('--use_saved_val', action='store_true', help='use saved validation problems')
    parser.add_argument('--save_path', type=str, default='image_robustfill_baseline0.p')
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--save_old_freq', type=int, default=10000)
    parser.add_argument('--positional', action='store_true')
    parser.add_argument('--hr', type=float, default = 0.5)
    parser.add_argument('--gpu', type=int, default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    batchsize = args.batchsize


    line = '(lambda (#(lambda (lambda (tower_loopM $0 (lambda (lambda (moveHand 3 (reverseHand (tower_loopM $3 (lambda (lambda (moveHand 6 (3x1 $0)))) $0)))))))) 6 6 $0))' 

    #print(stringify(line))

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


    fe = r.recognitionModel.featureExtractor
    g = r.recognitionModel.generativeModel #is this right?
    tasks = list(r.taskSolutions.keys())
    frontiers = list(r.allFrontiers.values())
    nonEmptyFrontiers = [fs for fs in frontiers if fs.entries]

    extras = ['(', ')', 'lambda', '#'] + ['$'+str(i) for i in range(10)]


    #input_vocabularies = [list(string.printable[:-4]) + ['EOE'], string.printable[:-4]]
    target_vocabulary = [str(p) for _, _, p in g.productions] + extras


    #assert stringify(line)[3] in target_vocabulary

    m = Image_RobustFill(target_vocabulary=target_vocabulary)
    m.cuda()
    m.iter = 0

    batch = [getDatum() for _ in range(batchsize)]
    inputs, targets = zip(*batch)
    t = time.time()
    for i in range(int(args.num_training_programs/args.batchsize)):

        batch = [getDatum() for _ in range(batchsize)]
        inputs, targets = zip(*batch)

        score = m.optimiser_step(inputs,targets) #syntax or not, idk
        m.iter += 1

        print(f"total time: {time.time() - t}, total num ex processed: {(i+1)*batchsize}, avg time per ex: {(time.time() - t)/((i+1)*batchsize)}, score: {score}", flush=True)

        if i%args.save_freq==0:
            torch.save(m, args.save_path)
            print('saved model')
        if i%args.save_old_freq==0:
            torch.save(m, args.save_path+str(m.iter))
