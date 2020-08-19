#simpleEval.py
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

import time
import argparse
from dreamcoder.enumeration import *
from dreamcoder.grammar import *
import torch
import dill
from dreamcoder.valueHead import SampleDummyValueHead, TowerREPLValueHead
from dreamcoder.policyHead import RNNPolicyHead, REPLPolicyHead

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='towers3')
parser.add_argument('--modeltype', type=str, default='REPL')
parser.add_argument('--nPerGrad', type=int, default=4)
parser.add_argument('--nSamples', type=int, default=4)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
#imports
class HelmholtzEntry:
    def __init__(self, frontier, owner):
        self.request = frontier.task.request
        self.task = None
        self.programs = [e.program for e in frontier]
        #MAX CHANGED:
        self.frontier = Thunk(lambda: owner.replaceProgramsWithLikelihoodSummaries(frontier, keepExpr=owner.useValue))
        #self.frontier = frontier
        self.owner = owner

    def clear(self): self.task = None

    def calculateTask(self):
        assert self.task is None
        p = random.choice(self.programs)
        return self.owner.featureExtractor.taskOfProgram(p, self.request)

    def makeFrontier(self):
        assert self.task is not None
        #MAX CHANGED
        f = Frontier(self.frontier.force().entries,
                     task=self.task)
        # f = Frontier(self.frontier.entries,
        #              task=self.task)
        return f

torch.cuda.set_device(args.gpu)

sys.setrecursionlimit(20000)
ID = args.id
graph = ""
model = 'RNN'
#When model is "Sample", don't need policyOnly
path = f'experimentOutputs/{ID}PolicyOnly{model}_SRE=True{graph}.pickle'

model = args.modeltype
modelPath=f"experimentOutputs/{ID}PolicyOnly{model}.pickle_RecModelOnly"

def loadRecModel():
    print(path)
    with open(path, 'rb') as h:
        result = dill.load(h)

    if args.modeltype == 'REPL':
        with open(modelPath, 'rb') as h:
            recModel = torch.load(h)
    else: recModel = result.recognitionModel
    recModel.cuda()
    return result, recModel

def get_rl_loss(frontier, r, nSamples=4):
    r.valueHead.train()

    entry = frontier.sample()
    task = frontier.task
    tp = frontier.task.request
    fullProg = entry.program._fullProg

    from dreamcoder.zipper import sampleSingleStep,baseHoleOfType,findHoles

    h = baseHoleOfType(tp)
    zippers = findHoles(h, tp)
        
    lls = []
    traces = []
    for _ in range(nSamples):
        trace = []
        newOb = h
        newZippers = zippers
        while newZippers:
            #newOb, newZippers = sampleSingleStep(gS, newOb, tp, holeZippers=newZippers, maximumDepth=8)
            newOb, newZippers = r.policyHead.sampleSingleStep(task, g, newOb, tp, holeZippers=newZippers, maximumDepth=10 )
            trace.append(newOb)

            #value = rR.recognitionModel.valueHead.computeValue(newOb, task)
            #dt = time.time() - t
            #valueTimes.append( dt )
            #if newZippers: neuralValues.append(value)
            #RNNValue = rRNN.recognitionModel.valueHead.computeValue(newOb, task)
            #concreteValue = concreteHead.computeValue(newOb, task)
        traces.append(trace)
        logLikelihood = task.logLikelihood(newOb, None)
        lls.append(logLikelihood)   



    posTrace = [p for t, ll in zip(traces,lls) for p in t if ll==0 ]
    negTrace = [p for t, ll in zip(traces,lls) for p in t if ll<0 ]

    #import pdb; pdb.set_trace()
    #print(lls)

    success = any(ll==0 for ll in lls)

    return r.valueHead._valueLossFromTraces(posTrace, negTrace, task), success


def createValueHeadFromPolicy(r):
    if isinstance(r.valueHead, SampleDummyValueHead):
        if isinstance(r.policyHead, REPLPolicyHead):
            r.valueHead = r.policyHead.REPLHead
            r.valueHead.rl_iterations = 0
        elif isinstance(r.policyHead, RNNPolicyHead):
            r.valueHead = r.policyHead.RNNHead
            r.valueHead.rl_iterations = 0
    elif isinstance(r.valueHead, TowerREPLValueHead):
        pass
    elif isinstance(r.valueHead, RNNValueHead):
        pass
    # much more annoying for robustfill



result, r = loadRecModel() #todo

createValueHeadFromPolicy(r)

optimizer = torch.optim.Adam(r.parameters(), lr=0.001, eps=1e-3, amsgrad=True)

g = r.generativeModel 
requests = [frontier.request for frontier in result.allFrontiers]

t = time.time()
for i in range(16000):
    r.zero_grad()
    rl_loss = 0
    contrastive_loss = 0
    policy_loss = 0

    for j in range(args.nPerGrad):

        frontier = None
        while frontier is None: frontier = r.sampleHelmholtz(requests)
        for e in frontier: e.program._fullProg = e.program        
        #contrastive_loss = r.valueHead.valueLossFromFrontier(frontier, g) #we do or dont need this    
        Ls, success = get_rl_loss(frontier, r, nSamples=args.nSamples)
        rl_loss += Ls        
        policy_loss += r.policyHead.policyLossFromFrontier(frontier, g)

    (rl_loss + contrastive_loss + policy_loss).backward()
    optimizer.step()
    r.valueHead.rl_iterations += args.nPerGrad*args.nSamples

    if i%10==0 and i !=0 : 
        print(f"iteration: {i}, rl loss: {rl_loss.item()}, time: {(time.time() - t)/i}")

    if i%100==0:
        torch.save(r, modelPath+'rl')
        print('saved model')