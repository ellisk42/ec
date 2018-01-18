from enumeration import *
from fragmentGrammar import *
from grammar import *
from utilities import eprint

import torch
import torch.nn as nn
import torch.optim as optimization
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

import numpy as np

class RecognitionModel(nn.Module):
    def __init__(self, featureDimensionality, grammar):
        super(RecognitionModel, self).__init__()
        self.grammar = grammar

        H = 5
        self.l1 = nn.Linear(featureDimensionality, H)
        self.logVariable = nn.Linear(H,1)
        self.logProductions = nn.Linear(H, len(self.grammar))
    
    def forward(self, features):
        h = F.sigmoid(self.l1(features))
        return self.logVariable(h),\
            self.logProductions(h)

    def extractFeatures(self, tasks):
        return Variable(torch.from_numpy(np.array([ task.features for task in tasks ])).float())
    
    def logLikelihood(self, frontiers):
        features = self.extractFeatures([ frontier.task for frontier in frontiers ])
        variables, productions = self(features)
        l = 0
        for j,frontier in enumerate(frontiers):
            v,p = variables[j],productions[j]
            g = FragmentGrammar(v, [(p[k],t,program) for k,(_,t,program) in enumerate(self.grammar.productions) ])
            l += lse([g.closedLogLikelihood(frontier.task.request, entry.program)
                      for entry in frontier ])
        return l

    def train(self, frontiers, _ = None, steps = 10**3, lr = 0.001, topK = 1):
        # Torch sometimes segfaults in multithreaded mode...
        torch.set_num_threads(1)
        
        frontiers = [ frontier.topK(topK) for frontier in frontiers if not frontier.empty ]
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for i in range(steps):
            self.zero_grad()
            l = -self.logLikelihood(frontiers)/len(frontiers)
            if i%50 == 0:
                eprint("Epoch",i,"Loss",l.data[0])
            l.backward()
            optimizer.step()

    def enumerateFrontiers(self, frontierSize, tasks):
        from time import time

        # Torch sometimes segfaults in multithreaded mode...
        torch.set_num_threads(1)

        start = time()
        features = self.extractFeatures(tasks)
        variables, productions = self(features)
        grammars = {task: Grammar(variables.data[j][0],
                                  [ (productions.data[j][k],t,p)
                                    for k,(_,t,p) in enumerate(self.grammar.productions) ])
                    for j,task in enumerate(tasks) }
        eprint("Evaluated recognition model in %f seconds"%(time() - start))

        return callCompiled(enumerateFrontiers,
                            grammars, frontierSize, tasks)
