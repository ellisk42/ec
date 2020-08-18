#simpleEval.py
try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module


from dreamcoder.enumeration import *
from dreamcoder.grammar import *
import torch
import dill
from dreamcoder.valueHead import SampleDummyValueHead, TowerREPLValueHead
from dreamcoder.policyHead import RNNPolicyHead, REPLPolicyHead

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

ID = 'towers' + str(3)
graph = ""
model = 'RNN'
#When model is "Sample", don't need policyOnly
path = f'experimentOutputs/{ID}PolicyOnly{model}_SRE=True{graph}.pickle'

model = 'REPL'
modelPath=f"experimentOutputs/{ID}PolicyOnly{model}.pickle_RecModelOnly"

def loadRecModel():
    print(path)
    with open(path, 'rb') as h:
        result = dill.load(h)
    with open(modelPath, 'rb') as h:
        recModel = torch.load(h)

    return result, recModel

def get_rl_loss(frontier, r, nSamples=4):
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

    import pdb; pdb.set_trace()

    return r.valueHead._valueLossFromTraces(posTrace, negTrace, task)


def createValueHeadFromPolicy(r):
    if isinstance(r.valueHead, SampleDummyValueHead):
        if isinstance(r.policyHead, REPLPolicyHead):
            r.valueHead = r.policyHead.REPLHead
        elif isinstance(r.policyHead, RNNPolicyHead):
            r.valueHead = r.policyHead.RNNHead
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

for i in range(1000000):
    #frontier = r.sampleHelmholtz(requests) #this can return None
    frontier = None
    while frontier is None: frontier = r.sampleHelmholtz(requests)

    for e in frontier:
        e.program._fullProg = e.program
    # e = HelmholtzEntry(frontier,r)
    # frontier = [e]

    r.zero_grad()
    contrastive_loss = r.valueHead.valueLossFromFrontier(frontier, g) #we do or dont need this
    #rl_loss = 0
    rl_loss = get_rl_loss(frontier, r)
    (rl_loss + contrastive_loss).backward()
    optimizer.step()

    print("contrastive_loss", contrastive_loss)
    print("rl_loss", rl_loss)




    #prints

    #saves