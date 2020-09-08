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
from dreamcoder.valueHead import SimpleRNNValueHead, binary_cross_entropy, TowerREPLValueHead, SimpleModularValueHead
from dreamcoder.program import Index, Program
from dreamcoder.zipper import *
from dreamcoder.utilities import count_parameters
from dreamcoder.domains.rb.rbPrimitives import *
from dreamcoder.ROBUT import ButtonSeqError, CommitPrefixError, NoChangeError

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

    def enumSingleStep(self, task, g, sk, request, 
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
        supplyDist = { expr: dist[i].data.item() for i, expr in self.indexToProduction.items()}

        # for k, v in supplyDist.items():
        #     print(v, k)
        # print()

        newSk, newZippers = sampleOneStepFromHole(zipper, sk, request, g, maximumDepth, supplyDist=supplyDist)
        return newSk, newZippers

    def enumSingleStep(self, task, g, sk, request, 
                        holeZipper=None,
                        maximumDepth=4):

        dist = self._computeDist([sk], [holeZipper], task, g)
        dist = dist.squeeze(0)
        supplyDist = { expr: dist[i].data.item() for i, expr in self.indexToProduction.items()}
        yield from enumSingleStep(g, sk, request, holeZipper=holeZipper, maximumDepth=maximumDepth, supplyDist=supplyDist)

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
    def __init__(self, g, featureExtractor, H=512, maxVar=15, encodeTargetHole=True, canonicalOrdering=False):
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


        print("num of params in rnn policy model", count_parameters(self))

        if self.use_cuda: self.cuda()

    def cuda(self, device=None):
        self.use_cuda = True
        self.RNNHead.use_cuda = True
        self.featureExtractor.use_cuda = True
        self.featureExtractor.CUDA = True
        super(RNNPolicyHead, self).cuda(device=device)

    def cpu(self):
        self.use_cuda = False
        self.RNNHead.use_cuda = False
        self.featureExtractor.use_cuda = False
        self.featureExtractor.CUDA = False
        super(RNNPolicyHead, self).cpu()

    def _computeDist(self, sketches, zippers, task, g):
        #need raw dist, and then which are valid and which is correct ... 
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


class ModularPolicyHead(RNNPolicyHead):
    def __init__(self, g, featureExtractor, H=512, maxVar=15, encodeTargetHole=True, canonicalOrdering=False):
        super(ModularPolicyHead, self).__init__() #should have featureExtractor?
        self.RNNHead = SimpleModularValueHead(g, featureExtractor, H=self.H, encodeTargetHole=encodeTargetHole) #hack


class REPLPolicyHead(NeuralPolicyHead):
    """
    does not specify the target hole at all here
    """
    def __init__(self, g, featureExtractor, H, maxVar=15, encodeTargetHole=False, canonicalOrdering=False, noConcrete=False):
        super(REPLPolicyHead, self).__init__() #should have featureExtractor?
        assert not encodeTargetHole
        self.canonicalOrdering = canonicalOrdering
        self.use_cuda = torch.cuda.is_available()
        self.featureExtractor = featureExtractor
        self.H = H
        self.REPLHead = TowerREPLValueHead(g, featureExtractor, H=self.H, noConcrete=noConcrete) #hack #TODO
        self.noConcrete = noConcrete

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
        self.use_cuda = True
        self.REPLHead.use_cuda = True
        self.featureExtractor.use_cuda = True
        self.featureExtractor.CUDA = True
        super(REPLPolicyHead, self).cuda(device=device)

    def cpu(self):
        self.use_cuda = False
        self.REPLHead.use_cuda = False
        self.featureExtractor.use_cuda = False
        self.featureExtractor.CUDA = False
        super(REPLPolicyHead, self).cpu()

    def _computeDist(self, sketches, zippers, task, g):
        #need raw dist, and then which are valid and which is correct ... 
        features = self.featureExtractor.featuresOfTask(task)
        if features is None: return None, None
        features = features.unsqueeze(0)
        
        sketchEncodings = [self.REPLHead._computeSketchRepresentation(sk.betaNormalForm()) for sk in sketches]
        sketchEncodings = torch.stack(sketchEncodings, dim=0)
        features = self.featureExtractor.featuresOfTask(task)
        features = features.unsqueeze(0)
        x = features.expand(len(sketches), -1)
        features = torch.cat([sketchEncodings, x ], dim=1)
        dist = self.output(features)
        mask = self._buildMask(sketches, zippers, task, g)
        dist = dist + mask
        return dist

class DenseLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(DenseLayer, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        #self.activation 
    def forward(self, x):
        return self.linear(x).relu()

class DenseBlock(nn.Module):
    def __init__(self, num_layers, growth_rate, input_size, output_size):
        super(DenseBlock, self).__init__()

        modules = [DenseLayer(input_size, growth_rate)]
        for i in range(1, num_layers - 1):
            modules.append(DenseLayer(growth_rate * i + input_size, growth_rate))
        modules.append(DenseLayer(growth_rate * (num_layers - 1) + input_size, output_size))
        self.layers = nn.ModuleList(modules)

    def forward(self, x):
        inputs = [x]
        for layer in self.layers:
            output = layer(torch.cat(inputs, dim=-1))
            inputs.append(output)
        return inputs[-1]


class SimpleNM(nn.Module):
    def __init__(self, nArgs, H=128):
        super(SimpleNM, self).__init__()
        self.nArgs = nArgs
        if nArgs > 0: #TODO
            #can just do a stack I think ...
            self.params = nn.Sequential(nn.Linear(nArgs*H, H), nn.ReLU())
        else:
            self.params = nn.Parameter(torch.randn(1, H))
        
    def forward(self, *args):
        #print(self.operator.name)
        if self.nArgs == 0:
            return self.params
        else:
            #print(type(args))
            inp = args[0]
            return self.params(inp)

class RBREPLPolicyHead(NeuralPolicyHead):
    """
    does not specify the target hole at all here
    """
    def __init__(self, g, featureExtractor, H, maxVar=15, encodeTargetHole=False, canonicalOrdering=False, noConcrete=False):
        super(RBREPLPolicyHead, self).__init__() #should have featureExtractor?
        assert not encodeTargetHole
        self.noConcrete = noConcrete

        self.canonicalOrdering = canonicalOrdering
        self.use_cuda = torch.cuda.is_available()
        self.featureExtractor = featureExtractor
        self.H = H
        #self.REPLHead = RBREPLValueHead(g, featureExtractor, H=self.H) #hack #TODO

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
                nn.Linear(H, H),
                nn.ReLU(),
                nn.Linear(H, H),
                nn.ReLU(),
                nn.Linear(H, len(self.productionToIndex) ),
                nn.LogSoftmax(dim=1))

        self.lossFn = nn.NLLLoss(reduction='sum')
        #print("num of params in repl policyhead (includes unused rnn weights)", count_parameters(self))


        #modules:
        self.appendModule = SimpleNM(2, H)
        self.compareModule = SimpleNM(2, H)


        self.encodeSimpleHole = nn.ModuleDict()
        for tp in [tposition, tregex, tindex, tboundary, tdelimiter, ttype]:
            self.encodeSimpleHole['t' + tp.show(True)] = SimpleNM(0, H)

        self.fnModules = nn.ModuleDict()
        for _, _, p in g.productions:
            if not p.isPrimitive: continue
            if p.tp == arrow(texpression, texpression):
                if self.noConcrete: self.fnModules[p.name] = SimpleNM(1, H)
                continue
            nArgs = len(p.tp.functionArguments())
            if p.tp.functionArguments() and p.tp.functionArguments()[0].isArrow():
               nArgs -= 1 
            
            self.fnModules[p.name] = SimpleNM(nArgs, H)
    
        self.encodeExprHole = SimpleNM(1, H)
        self.toFinishMarker = SimpleNM(1, H)

        print("num of params in repl policy model", count_parameters(self))
        if self.use_cuda: self.cuda()

    def cuda(self, device=None):
        self.use_cuda = True
        #self.REPLHead.use_cuda = True
        self.featureExtractor.use_cuda = True
        self.featureExtractor.CUDA = True
        super(RBREPLPolicyHead, self).cuda(device=device)

    def cpu(self):
        self.use_cuda = False
        #self.REPLHead.use_cuda = False
        self.featureExtractor.use_cuda = False
        self.featureExtractor.CUDA = False
        super(RBREPLPolicyHead, self).cpu()

    def _computeDist(self, sketches, zippers, task, g):
        features = self._computeREPR(sketches, task, zippers)
        dist = self.output(features)
        mask = self._buildMask(sketches, zippers, task, g)
        dist = dist + mask
        return dist

    def _computeREPR(self, sketches, task, zippers=None, outputRep=None):
        if zippers==None: zippers = [None for _ in sketches] #hack, because we don't need zippers for value
        if outputRep is None:
            outputRep = self.featureExtractor.outputsRepresentation(task).unsqueeze(0)
        outputRep = outputRep.expand(len(sketches), -1, -1)
        #print("outrep shape", outputRep.shape)
        currentState = [self._buildCurrentState(sk, zp, task) for sk, zp in zip(sketches, zippers)] 
        #print("currentstate0", currentState[0].shape)
        currentState = torch.stack(currentState, dim=0)
        #print("currentstate", currentState.shape)
        features = self.compareModule( torch.cat( [currentState, outputRep], dim=-1 ) ) #TODO batch
        #right now policy and value use same comparator
        #if we want value and policy comparators to be different,
        #we can do that by changing the above line and the def of _distance in valuehead
        return features.max(1)[0]

    def _buildCurrentState(self, sketch, zipper, task):

        prevSk, scratchSk = self._seperatePrevAndScratch(sketch, task.request, noConcrete=self.noConcrete)

        if self.noConcrete:
            prevRep = self.getPrevEncodingConcrete(sketch, task, task.request)
        else: prevRep = self.getPrevEncoding(prevSk, task)
        scratchRep = self.encodeScratch(scratchSk, task)

        currentState = self.appendModule(torch.cat( [prevRep, scratchRep], dim=-1 )) #TODO batch
        #append and compare modules do a lot of work
        return currentState

    def encodeScratch(self, sk, task, is_inner=False):
        """
        assume that sk is an e -> e sketch with stuff unfinished

        if it's not an n prim, need to insert a first arg, which is input
        if it is an nPrim, 
        """
        if sk.isHole:
            inputRep = self.featureExtractor.inputsRepresentation(task) #TODO
            return self.encodeExprHole(inputRep)

        f, xs = sk.applicationParse()
        isNPrim = (f.tp.functionArguments() and f.tp.functionArguments()[0] == arrow(texpression, texpression))

        #might be okay if is_inner triggers nothing

        args = xs[:-1]
        neuralArgs = []

        #deal with tricky first arg 
        if not isNPrim:
            inputRep = self.featureExtractor.inputsRepresentation(task) #TODO
            neuralArgs.append(inputRep)
        else:
            inputRep = self.encodeInnerN(args[0], findHoles(args[0], task.request), task) #TODO #args[0] could be a full hole ig..
            neuralArgs.append(inputRep)
            args = args[1:]

        #deal with rest of args
        for arg in args:
            if arg.isHole:
                neuralArgs.append( self.encodeSimpleHole['t' + arg.tp.show(True)]().expand(4, -1) ) #TODO
            else:
                neuralArgs.append( self.fnModules[arg.name]().expand(4, -1) ) #TODO

        fn_input = torch.cat(neuralArgs, dim=-1)
        return self.fnModules[f.name](fn_input) #TODO


    def encodeInnerN(self, sk, zippers, task):
        assert sk.isAbstraction
        inputRep = self.featureExtractor.inputsRepresentation(task)

        #if unchosen:
        if sk.body.isHole:
            return self.encodeExprHole(inputRep)
        if not zippers:
            return self.getPrevEncoding(sk, task)

        if len(zippers)==1: 
            assert zippers[0].tp == texpression
            completedSk = NewExprPlacer().execute(sk, zippers[0].path, Index(0))
            return self.toFinishMarker(self.getPrevEncoding(completedSk, task)) #TODO

        return self.encodeScratch(sk.body, task, is_inner=True)

        #big q: how do i encode that we need to chose to end the inner fn???
        #can just include an unfinishedParam, which we need to put in ... oy.

    def getPrevEncoding(self, sk, task):
        I, O = zip(*task.examples)
        exprs = sk.evaluate([])
        try:
            newP = ROB.P( exprs ([]) ) 
            previousWords = [ ROB.executeProg(newP, i) for i in I]
            return self.featureExtractor.encodeString(previousWords) #TODO

        # except  as e:
        #     previousWords = [i for i in I ]
        #     return self.featureExtractor.encodeString(previousWords)

        except (IndexError, NoChangeError) as e:
            # print(e)
            # previousWords = [i for i in I ] #Terrible hack ... you are right.
            # print(sk)
            # print(task)
            assert False, "this should have been taken care of by valueHead"
            return self.featureExtractor.encodeString(previousWords) #TODO

    def getPrevEncodingConcrete(self, sk, task, request):

        prev, scratchSk = self._seperatePrevAndScratchNoConcrete(sk, request)
        if prev == baseHoleOfType(request): #TODO:
            return self.encodeScratch(scratchSk, task)

        prevRep = self.getPrevEncodingConcrete(prev, task, request)
        scratchRep = self.encodeScratch(scratchSk, task)
        return self.appendModule(torch.cat( [prevRep, scratchRep], dim=-1 )) #TODO batch

    def _seperatePrevAndScratch(self, sk, request, noConcrete=False):
        """
        prev should be full prog
        scratch is not
        """
        zippers = findHoles(sk, request)
        if len(zippers) == 1:
            assert zippers[0].tp == texpression
            scratch = Hole(tp=texpression)
            newExpr = Hole(tp=texpression) if noConcrete else Index(0)
            prev = NewExprPlacer().execute(sk, zippers[0].path, newExpr)

        else: 
            commonPath = []
            for group in zip(*[zipp.path for zipp in zippers]):
                if all(move == group[0] for move in group ):
                    commonPath.append(group[0])
                else: break
            newExpr = Hole(tp=texpression) if noConcrete else Index(0) 
            prev, scratch = NewExprPlacer(allowReplaceApp=True, returnInnerObj=True ).execute(sk, commonPath, newExpr) 

        return prev, scratch


    def _seperatePrevAndScratchNoConcrete(self, sk, request):
        """
        prev should be full prog
        scratch is not
        """
        if sk == baseHoleOfType(request):
            return sk, Hole(tp=texpression)

        zippers = findHoles(sk, request)
        commonPath = zippers[0].path[:-1]
        print("Cpath", commonPath)

        print(sk)
        prev, scratch = NewExprPlacer(allowReplaceApp=True, returnInnerObj=True ).execute(sk, commonPath, Hole(tp=texpression)) 
        print(prev)
        print(scratch)

        return prev, scratch




