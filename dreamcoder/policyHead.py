#policyHead

"""
if useValue is on, then require a policyHead as well, and specify which type
PolicyHead

solver needs to know about policyHead

policy needs to know how to get g ?

policy is called exclusively at (train time) or by solver, so solver can shove itself in there for base policy ... 

recognition.py:
- [X] init: should take policyHead to construct
- [X] train: if useValue: policyHead.computePolicyLoss always
- [X] inference: .cpu and .cuda policyHead

dreamcoder.py:
- [X] constructor for policyHead
- [X] singleValround thing

Astar.py
- [X] rewrite so it calls the policyHead

SMC
- [X] rewrite so it calls the policyHead

- [X] supply dist
    - [X] zipper
    - [X] grammar

- [X] specialHole ...
Build simplest policyHead

- [X] deal with token specific hole
- [X] canonical ordering

What to do about grammar we infer? leave it in ...

REPL HEAD:
- [X] build tower repl policy

- [X] for RNN, do encodeTarget option
- [X] do canonicalOrderingOption

- [ ] make easy way to copy weights
- [ ] fuss with featureExtractor

- [ ] do light policy training

- [ ] test on frontiers w/out doing search


"""
import torch
from torch import nn

from dreamcoder.zipper import sampleSingleStep, enumSingleStep
from dreamcoder.valueHead import SimpleRNNValueHead, binary_cross_entropy, TowerREPLValueHead
from dreamcoder.program import Index, Program
from dreamcoder.zipper import *

class BasePolicyHead(nn.Module):
    #this is the single step type
    def __init__(self):
        super(BasePolicyHead, self).__init__() #should have featureExtractor?
        self.use_cuda = torch.cuda.is_available()

    def sampleSingleStep(self, task, g, sk,
                        request, holeZippers=None,
                        maximumDepth=4):
        return sampleSingleStep(g, sk, request, holeZippers=holeZippers, maximumDepth=maximumDepth)

    def policyLossFromFrontier(self, frontier, g):
        if self.use_cuda:
            return torch.tensor([0.]).cuda()
        else: 
            return torch.tensor([0.])

    def enumSingleStep(g, sk, request, 
                        holeZipper=None,
                        maximumDepth=4):
        return enumSingleStep(g, sk, request, holeZipper=holeZipper, maximumDepth=maximumDepth)


class NeuralPolicyHead(nn.Module):
    def __init__(self):
        super(NeuralPolicyHead, self).__init__()

    def sampleSingleStep(self, task, g, sk,
                        request, holeZippers=None,
                        maximumDepth=4):

        if self.canonicalOrdering: zipper = holeZippers[0]
        else: zipper = random.choice(holeZippers)
        dist = self._computeDist([sk], [zipper], task, g) #TODO
        dist = dist.squeeze(0)
        supplyDist = { expr: dist[self.productionToIndex[expr]].data.item() for _, _, expr in g.productions}
        newSk, newZippers = sampleOneStepFromHole(zipper, sk, request, g, maximumDepth, supplyDist=supplyDist)
        return newSk, newZippers

    def enumSingleStep(g, sk, request, 
                        holeZipper=None,
                        maximumDepth=4):

        dist = computeDist([sk], [holeZipper], task, g)
        supplyDist = { expr: dist[self.productionToIndex[expr]].data.item() for _, _, expr in g.productions}
        yield from enumSingleStep(g, sk, tp, holeZipper=holeZipper, maximumDepth=maximumDepth, supplyDist=supplyDist)

    def policyLossFromFrontier(self, frontier, g):
        # Monte Carlo estimate: draw a sample from the frontier
        entry = frontier.sample()
        tp = frontier.task.request
        
        if isinstance(entry.program, Program):
            fullProg = entry.program
        else:
            fullProg = entry.program._fullProg

        posTraces, _, targetNodes, holesToExpand = getTracesFromProg(fullProg, frontier.task.request, g, 
                                                        onlyPos=True, returnNextNode=True,
                                                        canonicalOrdering=self.canonicalOrdering)
        maskedDist = self._computeDist(posTraces, holesToExpand, frontier.task, g) #TODO

        targets = [self._sketchNodeToIndex(node) for node in targetNodes]
        targets = torch.tensor(targets)
        if self.use_cuda:
            targets = targets.cuda()
        loss = self.lossFn(maskedDist, targets)
        return loss

    def _computeDist(): raise NotImplementedError

    def _designateTargetHole(self, zipper, sk):
        specialHole = Hole(target=True)
        newSk = NewExprPlacer().execute(sk, zipper.path, specialHole) #TODO
        return newSk

    def _sketchNodeToIndex(self, node):
        if node in self.productionToIndex:
            return self.productionToIndex[node]
        if node.isAbstraction:
            return self._sketchNodeToIndex(node.body)
        if node.isApplication:
            f, xs = node.applicationParse()
            return self._sketchNodeToIndex(f)            
        else: assert False, f"invalid node {node}"

    def _buildMask(self, sketches, zippers, task, g):
        masks = []
        for zipper, sk in zip(zippers, sketches):
            mask = [0. for _ in range(len(self.productionToIndex))]
            candidates = returnCandidates(zipper, sk, task.request, g)
            for c in candidates:
                mask[self._sketchNodeToIndex(c)] = 1. 
            masks.append(mask)
        mask = torch.tensor(masks)
        mask = torch.log(mask)
        if self.use_cuda: mask = mask.cuda()
        return mask


class RNNPolicyHead(NeuralPolicyHead):
    def __init__(self, g, featureExtractor, H, maxVar=15, encodeTargetHole=True, canonicalOrdering=False):
        super(RNNPolicyHead, self).__init__() #should have featureExtractor?
        self.use_cuda = torch.cuda.is_available()
        self.featureExtractor = featureExtractor
        self.H = H
        self.RNNHead = SimpleRNNValueHead(g, featureExtractor, H=self.H, encodeTargetHole=encodeTargetHole) #hack
        self.encodeTargetHole = encodeTargetHole
        self.canonicalOrdering = canonicalOrdering

        self.indexToProduction = {}
        self.productionToIndex = {}
        i = 0
        for _, _, expr in g.productions:
            self.indexToProduction[i] = expr
            self.productionToIndex[expr] = i
            i += 1

        for v in range(maxVar):
            self.indexToProduction[i] = Index(v)
            self.productionToIndex[Index(v)] = i
            i += 1

        self.output = nn.Sequential(
                nn.Linear(featureExtractor.outputDimensionality + H, H),
                nn.ReLU(),
                nn.Linear(H, H),
                nn.ReLU(),
                nn.Linear(H, len(self.productionToIndex) ),
                nn.LogSoftmax(dim=1))

        self.lossFn = nn.NLLLoss(reduction='sum')
        if self.use_cuda: self.cuda()

    def cuda(self, device=None):
        self.RNNHead.use_cuda = True
        self.featureExtractor.use_cuda = True
        self.featureExtractor.CUDA = True
        super(RNNPolicyHead, self).cuda(device=device)

    def cpu(self):
        self.RNNHead.use_cuda = False
        self.featureExtractor.use_cuda = False
        self.featureExtractor.CUDA = False
        super(RNNPolicyHead, self).cpu()

    def _computeDist(self, sketches, zippers, task, g):
        #need raw dist, and then which are valid and which is correct ... 
        features = self.featureExtractor.featuresOfTask(task)
        if features is None: return None, None
        features = features.unsqueeze(0)
        if self.encodeTargetHole:
            sketches = [self._designateTargetHole(zipper, sk) for zipper, sk in zip(zippers, sketches)]
        sketchEncodings = self.RNNHead._encodeSketches(sketches)
        features = self.featureExtractor.featuresOfTask(task)
        features = features.unsqueeze(0)
        x = features.expand(len(sketches), -1)
        features = torch.cat([sketchEncodings, x ], dim=1)
        dist = self.output(features)
        mask = self._buildMask(sketches, zippers, task, g)
        dist = dist + mask
        return dist


class REPLPolicyHead(NeuralPolicyHead):
    """
    does not specify the target hole at all here
    """
    def __init__(self, g, featureExtractor, H, maxVar=15, encodeTargetHole=False, canonicalOrdering=False):
        super(REPLPolicyHead, self).__init__() #should have featureExtractor?
        assert not encodeTargetHole
        self.canonicalOrdering = canonicalOrdering
        self.use_cuda = torch.cuda.is_available()
        self.featureExtractor = featureExtractor
        self.H = H
        self.REPLHead = TowerREPLValueHead(g, featureExtractor, H=self.H) #hack #TODO

        self.indexToProduction = {}
        self.productionToIndex = {}
        i = 0
        for _, _, expr in g.productions:
            self.indexToProduction[i] = expr
            self.productionToIndex[expr] = i
            i += 1

        for v in range(maxVar):
            self.indexToProduction[i] = Index(v)
            self.productionToIndex[Index(v)] = i
            i += 1

        self.output = nn.Sequential(
                nn.Linear(featureExtractor.outputDimensionality + H, H),
                nn.ReLU(),
                nn.Linear(H, H),
                nn.ReLU(),
                nn.Linear(H, len(self.productionToIndex) ),
                nn.LogSoftmax(dim=1))

        self.lossFn = nn.NLLLoss(reduction='sum')
        if self.use_cuda: self.cuda()

    def cuda(self, device=None):
        self.REPLHead.use_cuda = True
        self.featureExtractor.use_cuda = True
        self.featureExtractor.CUDA = True
        super(REPLPolicyHead, self).cuda(device=device)

    def cpu(self):
        self.REPLHead.use_cuda = False
        self.featureExtractor.use_cuda = False
        self.featureExtractor.CUDA = False
        super(REPLPolicyHead, self).cpu()

    def _computeDist(self, sketches, zippers, task, g):
        #need raw dist, and then which are valid and which is correct ... 
        features = self.featureExtractor.featuresOfTask(task)
        if features is None: return None, None
        features = features.unsqueeze(0)
        
        sketchEncodings = [self.REPLHead._computeSketchRepresentation(sk) for sk in sketches]
        sketchEncodings = torch.stack(sketchEncodings, dim=0)
        features = self.featureExtractor.featuresOfTask(task)
        features = features.unsqueeze(0)
        x = features.expand(len(sketches), -1)
        features = torch.cat([sketchEncodings, x ], dim=1)
        dist = self.output(features)
        mask = self._buildMask(sketches, zippers, task, g)
        dist = dist + mask
        return dist
